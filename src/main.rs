use std::env;

use std::sync::Arc;
use std::io::{self, Write};
use chrono::{DateTime, Utc};
use serde::Deserialize;

use serde_json::Value as JsonValue;
use langgraph::checkpoint::InMemorySaver;
use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph::tool;
use langgraph::langgraph_state;
use langgraph::prebuilt::{BaseChatModel, Message, ToolNode, prepare_tools, print_stream,stream_llm,tools_condition};
use langgraph::providers::openai::{OpenAIModelConfig,OpenAIModel};


//定义state
#[langgraph_state]
#[derive(Debug)]
struct GraphState{
    #[channel(messages)]
    messages: Vec<Message>,
    #[channel]
    search_context: String,
}

//定义搜索记忆的返回结构体
#[derive(Debug,Deserialize)]
struct Memory {
    id: Option<String>,
    content: String,
    created_at: Option<String>,
    updated_at: Option<String>,
}
#[derive(Debug, Deserialize)]
struct SearchResponse {
    memories: Vec<Memory>,
}


//辅助处理时间的函数
fn parse_time_to_chinese_date(iso_str: &str) -> String {
    // 尝试解析带时区（含 Z）的时间
    if let Ok(dt) = DateTime::parse_from_rfc3339(iso_str) {
        // 转为 UTC 时间，再格式化
        let utc = dt.with_timezone(&Utc);
        return utc.format("%Y年%m月%d日").to_string();
    }
    // 如果解析失败，截取前 10 个字符（如 "2026-07-11"）
    iso_str.chars().take(10).collect()
}


//定义工具函数
#[tool("search_memory"," 从记忆中深度搜索相关的历史信息和对话。当系统未给出合适的信息时使用。
    当用户询问'你还记得吗'、'之前说过'、'上次'、'以前'、'有没有'、'记不记得'等涉及过去事件的问题时必须使用此工具！
    也可用于主动搜索用户的偏好、经历、约定等
    参数：query: 需要查询的内容")]
async fn search_memory(query: String) -> Result<String, String> {
    let base_url = env::var("MEMOS_BASE_URL")
        .map_err(|_| "MEMOS_BASE_URL not set".to_string())?;
    let user_id = env::var("USER_ID")
        .unwrap_or_else(|_| "default".to_string());

    let client = reqwest::Client::new();
    let payload = serde_json::json!({
        "query": query,
        "user_id": user_id,
        "top_k": 3,
        "similarity_threshold": 0.5
    });

    let url = format!("{}/search", base_url);
    let response = client
        .post(&url)
        .json(&payload)
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .map_err(|e| format!("请求失败: {}", e))?;

    // 先获取状态码
    let status = response.status();
    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!("搜索失败 (HTTP {}): {}", status, error_text));
    }

    let data: SearchResponse = response
        .json()
        .await
        .map_err(|e| format!("解析失败: {}", e))?;

    if data.memories.is_empty() {
        return Ok(format!("没有与'{}'相关的记忆", query));
    }

    let mut lines = Vec::new();
    for mem in data.memories {
        let mut line = mem.content;

        // 使用引用避免移动
        if let Some(created) = &mem.created_at {
            let time_str = parse_time_to_chinese_date(created);
            line.push_str(&format!(" [{}]", time_str));
        }

        if let Some(updated) = &mem.updated_at {
            if mem.created_at.as_ref() != Some(updated) {
                line.push_str(" (已更新)");
            }
        }

        // 使用引用
        if let Some(id) = &mem.id {
            if !id.is_empty() {
                line.push_str(&format!(" [ID: {}]", id));
            }
        }

        lines.push(line);
    }

    Ok(format!("相关记忆:\n{}", lines.join("\n")))
}


//从环境变量中加载配置
fn load_openai_config() -> (String, Option<String>, String) {
    dotenv().ok();
    let api_key =
        std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env or environment");
    let api_base = std::env::var("OPENAI_API_BASE").ok();
    let model_name = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "mimo-v2.5-pro".to_string());

    (api_key, api_base, model_name)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //初始化模型
    let (api_key, api_base, model_name) = load_openai_config();
    let model = OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        ..Default::default()
    });

    //准备工具
    let prepared = prepare_tools(vec![
        Arc::new(SearchMemory::new()),
    ]);
    let model_with_tools: Arc<dyn BaseChatModel> = model.bind_tools(prepared.tool_defs).into();

    let channels = GraphState::create_channels();
    let mut graph = StateGraph::new(channels);

    graph.add_node("search_memories", move |input: JsonValue, _config: RunnableConfig| {
    async move {
        // 提取用户消息
        let last_user_msg = input
            .get("messages")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.last())
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");

        if last_user_msg.is_empty() {
            return Ok(serde_json::json!({"search_context": ""}));
        }

        // 读取环境变量
        let base_url = match env::var("MEMOS_BASE_URL") {
            Ok(v) => v,
            Err(_) => return Err(RunnableError::Other("MEMOS_BASE_URL not set".into())),
        };
        let user_id = env::var("USER_ID")
           .unwrap_or_else(|_| "01".to_string());

        //构造请求
        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "query": last_user_msg,
            "user_id": user_id,
            "top_k": 3,
            "similarity_threshold": 0.5
        });

        let url = format!("{}/search", base_url);

        // 4. 发送请求（显式 match 处理错误）
        let response = match client
            .post(&url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return Err(RunnableError::Other(Box::new(e))),
        };

        //检查状态码并解析
        let context = if response.status().is_success() {
            // 使用 SearchResponse 结构体解析
            let data: SearchResponse = match response.json().await {
                Ok(d) => d,
                Err(e) => return Err(RunnableError::Other(Box::new(e))),
            };
            // 只提取 content，换行拼接
            data.memories
                .into_iter()
                .map(|m| m.content)
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            eprintln!("搜索失败，状态码: {}", response.status());
            String::new()
        };

        // 6. 返回状态更新
        Ok(serde_json::json!({"search_context": context}))
    }
    })?;

    let model_clone = model_with_tools.clone();
    graph.add_node("llm_call", move |input: JsonValue, _config: RunnableConfig| {
    let model = model_clone.clone();
    async move {
        let context = input.get("search_context").and_then(|c| c.as_str()).unwrap_or("");
        let mut sys_prompt = "你是一个助手".to_string();
        if !context.is_empty() {
            sys_prompt.push_str(&format!("\n以下是可能相关的记忆：\n{}", context));
        }
        // 注意：stream_llm 可能不支持工具调用，如果你仍想保留工具调用，需要改用 invoke
        let result = stream_llm(
            model.as_ref(),
            &input,
            &sys_prompt,
        ).await?;
        // 假设 stream_llm 返回的是包含 messages 的 JsonValue
        return Ok(result);
    }
    })?;

    //添加工具节点
    let tool_node: Arc<dyn Runnable> = Arc::new(ToolNode::new(prepared.tools.clone()));
    graph.add_node("tool_node", tool_node)?;

    //组装图
    graph.add_edge(START, "search_memories")?;
    graph.add_edge("search_memories", "llm_call")?;
    conditional_edges!(graph,"llm_call",tools_condition,"tools"=>"tool_node",END => END)?;
    graph.add_edge("tool_node", "llm_call")?;
    let checkpointer = Arc::new(InMemorySaver::new());

    //编译图
    let agentgraph = graph.compile_builder().checkpointer(checkpointer).build()?;

    let mut config = RunnableConfig::new();
    config.insert(
        "configurable".to_string(),
        serde_json::json!({"thread_id": "interactive-session"}),
    );

    let stdin = io::stdin();
    loop {
        print!("You: ");
        io::stdout().flush()?;
        let mut input_line = String::new();
        if stdin.read_line(&mut input_line)? == 0 {
            break;
        }
        let input_line = input_line.trim();

        if input_line.eq_ignore_ascii_case("quit") || input_line.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        if input_line.is_empty() {
            continue;
        }

        let input = serde_json::json!({
            "messages": [{"type": "human", "content": input_line}]
        });

        // 启动流式执行
        let mut stream = agentgraph.astream(
            &input,
            &config,
            vec![StreamMode::Custom, StreamMode::Updates],
        );

        print!("Assistant: ");
        io::stdout().flush()?;

        // 使用 print_stream 处理所有事件（包括 token 和更新）
        // 第二个参数设为 true 可能显示详细信息，设为 false 只显示 token
        let _ = print_stream(&mut stream, false).await;

        println!(); // 换行
    }

    return Ok(());
}