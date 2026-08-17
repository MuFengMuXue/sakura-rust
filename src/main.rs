use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::env;
use std::io::{self, Write};
use std::sync::Arc;

use dotenvy::dotenv;
use langgraph::checkpoint::InMemorySaver;
use langgraph::langgraph_state;
use langgraph::prebuilt::{
    BaseChatModel, Message, ToolNode, prepare_tools, print_stream, stream_llm, tools_condition,
};
use langgraph::prelude::*;
use langgraph::providers::openai::{OpenAIModel, OpenAIModelConfig};
use langgraph::tool;
use serde_json::Value as JsonValue;

//定义state
#[langgraph_state]
#[derive(Debug)]
struct GraphState {
    #[channel(messages)]
    messages: Vec<Message>,
}

///搜索记忆
//定义搜索记忆的请求结构体（发送给/search）
#[derive(Serialize)]
struct SearchRequest {
    user_id: String,
    query: String,
    top_k: u8,                 // 或 i32
    similarity_threshold: f32, // 或 f32
}

//搜索记忆的返回结构体
#[derive(Deserialize)]
struct SearchMessage {
    id: Option<String>,
    content: String,
    created_at: Option<String>,
    updated_at: Option<String>,
}
#[derive(Deserialize)]
struct SearchResponse {
    memories: Vec<SearchMessage>,
}

///反馈记忆
// 反馈请求结构（发送给 /memory/feedback）
#[derive(Serialize)]
struct FeedbackRequest {
    memory_id: String,
    feedback_type: String, // correct, supplement, delete
    reason: String,
    user_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    correction: Option<String>,
}
// 反馈响应结构（从服务端解析）
#[derive(Deserialize)]
struct FeedbackResponse {
    status: String,
    #[serde(default)]
    new_content: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

///添加记忆
//添加记忆请求结构(发送给/add)
#[derive(Serialize)]
struct AddMessage {
    role: String,
    content: String,
}
#[derive(Serialize)]
struct AddRequest {
    messages: Vec<AddMessage>,
    user_id: String,
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
    return iso_str.chars().take(10).collect();
}

//定义工具函数
#[tool("search_memory"," 从记忆中深度搜索相关的历史信息和对话。当系统未给出合适的信息时使用。
    当用户询问'你还记得吗'、'之前说过'、'上次'、'以前'、'有没有'、'记不记得'等涉及过去事件的问题时必须使用此工具！
    也可用于主动搜索用户的偏好、经历、约定等
    参数：query: 需要查询的内容")]
async fn search_memory(query: String) -> Result<String, String> {
    let base_url = env::var("MEMOS_BASE_URL").map_err(|_| "MEMOS_BASE_URL not set".to_string())?;
    let user_id = env::var("USER_ID").unwrap_or_else(|_| "default".to_string());
    let url = format!("{}/search", base_url);

    let client = reqwest::Client::new();
    let payload = SearchRequest {
        query: query.clone(),
        user_id: user_id,
        top_k: 3,
        similarity_threshold: 0.5,
    };

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
        return Ok(format!("没有与{}相关的记忆", query));
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

#[tool(
    "correct_memory",
    "修正、补充或删除已有的记忆。
    需要先用 memos_search_memory 获取记忆 ID。
    参数：memory_id: 要操作的记忆 ID（通过搜索获取）
         action: 操作类型，可选值：'correct'（修正）、'supplement'（补充）、'delete'（删除）
         new_content: 修正或补充的新内容（当 action 为 correct 或 supplement 时必填）
         reason: 操作原因"
)]
async fn correct_memory(
    memory_id: String,
    action: String,
    new_content: Option<String>,
    reason: Option<String>,
) -> Result<String, String> {
    //参数检验
    if memory_id.is_empty() {
        return Err("错误，未提供记忆id。请先通过search_memory获取id".to_string());
    }
    let action_str = action.trim().to_lowercase();
    if !["correct", "supplement", "delete"].contains(&action_str.as_str()) {
        return Err("错误：未指定操作类型。可选：correct、supplement、delete".to_string());
    }
    if (action_str == "correct" || action_str == "supplement")
        && new_content.as_ref().map_or(true, |s| s.is_empty())
    {
        return Err(format!("错误：{} 必须提供 new_content", action_str));
    }

    //读取环境变量
    let base_url = env::var("MEMOS_BASE_URL").map_err(|_| "MEMOS_BASE_URL not set".to_string())?;
    let user_id = env::var("USER_ID").unwrap_or_else(|_| "default".to_string());
    let url = format!("{}/memory/feedback", base_url);

    //构造请求
    let client = reqwest::Client::new();
    let payload = FeedbackRequest {
        memory_id: memory_id,
        user_id: user_id,
        reason: reason.unwrap_or_default(),
        feedback_type: action_str.clone(),
        correction: if action_str == "correct" || action_str == "supplement" {
            new_content
        } else {
            None
        },
    };

    //发送请求
    let response = client
        .post(&url)
        .json(&payload)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("请求失败: {}", e))?;

    //处理响应状态码
    let status = response.status();
    if status == 200 {
        let data: FeedbackResponse = response
            .json()
            .await
            .map_err(|e| format!("解析响应失败: {}", e))?;

        if data.status == "success" {
            if action_str == "delete" {
                return Ok(format!("记忆成功删除，ID: {}", payload.memory_id));
            } else {
                let action_name = if action_str == "correct" {
                    "修正"
                } else {
                    "补充"
                };
                let new_content_display = data.new_content.unwrap_or_else(|| {
                    // 如果服务端没返回新内容，使用请求中的
                    payload.correction.unwrap_or_default()
                });
                return Ok(format!(
                    "已成功{}\nID: {}\n新内容: {}",
                    action_name, payload.memory_id, new_content_display
                ));
            }
        } else {
            let msg = data.message.unwrap_or_else(|| "未知错误".to_string());
            return Err(format!("操作失败：{}", msg));
        }
    } else if status == 404 {
        return Err(format!(
            "记忆ID {} 不存在，请确认ID是否正确",
            payload.memory_id
        ));
    } else {
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!("操作失败 (HTTP {}): {}", status, error_text));
    }
}

#[tool(
    "add_memory",
    " 记住重要信息。
    当用户明确说“记住”、“别忘了”，或者透露个人偏好时使用。
    或者是你觉得重要的内容，也可以调用。
    参数：
        content: 要记住的内容（简洁明了）"
)]
async fn add_memory(content: String) -> Result<String, String> {
    //读取环境变量
    let base_url = env::var("MEMOS_BASE_URL").map_err(|_| "MEMOS_BASE_URL not set".to_string())?;
    let user_id = env::var("USER_ID").unwrap_or_else(|_| "default".to_string());
    let url = format!("{}/add", base_url);

    //构造请求
    let client = reqwest::Client::new();
    let payload = AddRequest {
        messages: vec![AddMessage {
            role: "user".to_string(),
            content: content.clone(),
        }],
        user_id,
    };

    //发送请求
    let response = client
        .post(&url)
        .json(&payload)
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .map_err(|e| format!("请求失败: {}", e))?;

    let status = response.status();
    if status == 200 {
        return Ok(format!("已成功记住{}", content));
    } else {
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!(
            "出现错误，状态码：{}，具体信息：{}",
            status, error_text
        ));
    }
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
    env_logger::init();
    //初始化模型
    let (api_key, api_base, model_name) = load_openai_config();
    let model = OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        extra_body: Some(serde_json::json!({
            "thinking": {"type": "disabled"},
        })),
        ..Default::default()
    });

    //准备工具
    let prepared = prepare_tools(vec![
        Arc::new(SearchMemory::new()),
        Arc::new(CorrectMemory::new()),
        Arc::new(AddMemory::new()),
    ]);
    let model_with_tools: Arc<dyn BaseChatModel> = model.bind_tools(prepared.tool_defs).into();

    // 被动注入去重窗口:近 N 轮内已注入过的记忆不重复注入;0=全历史去重;缺失/非法=10
    let dedup_window: usize = env::var("DEDUP_WINDOW_TURNS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(10);

    let channels = GraphState::create_channels();
    let mut graph = StateGraph::new(channels);

    graph.add_node(
        "search_memories",
        move |input: JsonValue, _config: RunnableConfig| {
            let window = dedup_window;
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
                    return Ok(serde_json::json!({}));
                }

                // 读取环境变量
                let base_url = match env::var("MEMOS_BASE_URL") {
                    Ok(v) => v,
                    Err(_) => return Err(RunnableError::Other("MEMOS_BASE_URL not set".into())),
                };
                let user_id = env::var("USER_ID").unwrap_or_else(|_| "default".to_string());

                //构造请求
                let client = reqwest::Client::new();
                let payload = SearchRequest {
                    query: last_user_msg.to_string(),
                    user_id: user_id,
                    top_k: 5,
                    similarity_threshold: 0.4,
                };

                let url = format!("{}/search", base_url);

                //发送请求（显式 match 处理错误）
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

                //检查状态码并解析,按整条记忆收集(不做行拼接,保留多行内容)
                let entries: Vec<String> = if response.status().is_success() {
                    let data: SearchResponse = match response.json().await {
                        Ok(d) => d,
                        Err(e) => return Err(RunnableError::Other(Box::new(e))),
                    };
                    data.memories.into_iter().map(|m| m.content).collect()
                } else {
                    eprintln!("搜索失败，状态码: {}", response.status());
                    Vec::new()
                };

                // ── 近 N 轮窗口去重 ──
                let msgs = input.get("messages").and_then(|m| m.as_array());

                // 判据:上下文块(带固定标记)vs 真·用户回合
                let is_block = |m: &JsonValue| {
                    m.get("content")
                        .and_then(|c| c.as_str())
                        .map_or(false, |c| c.starts_with("<search-context>"))
                };
                let is_real_human = |m: &JsonValue| {
                    m.get("type").and_then(|t| t.as_str()) == Some("human") && !is_block(m)
                };

                let n_humans = msgs
                    .map(|arr| arr.iter().filter(|m| is_real_human(m)).count())
                    .unwrap_or(0);

                // 窗口起点:第 (n_humans - window) 个真回合之后;window==0 或回合不足则全历史
                let mut cutoff = 0usize;
                if window > 0 && n_humans > window {
                    if let Some(arr) = msgs {
                        let mut seen_turns = 0usize;
                        for (i, m) in arr.iter().enumerate() {
                            if is_real_human(m) {
                                seen_turns += 1;
                            }
                            if seen_turns == n_humans - window {
                                cutoff = i + 1;
                                break;
                            }
                        }
                    }
                }

                // 收集窗口内已注入过的条目文本:块内 "- " 开头开新条目,其余行并入上一条
                let mut seen: HashSet<String> = HashSet::new();
                if let Some(arr) = msgs {
                    for m in arr.iter().skip(cutoff) {
                        let content = match m.get("content").and_then(|c| c.as_str()) {
                            Some(c) if c.starts_with("<search-context>") => c,
                            _ => continue,
                        };
                        let mut current: Vec<&str> = Vec::new();
                        for line in content.lines() {
                            if line == "</search-context>" {
                                if !current.is_empty() {
                                    seen.insert(current.join("\n").trim().to_string());
                                }
                                current.clear();
                                break;
                            }
                            if let Some(rest) = line.strip_prefix("- ") {
                                if !current.is_empty() {
                                    seen.insert(current.join("\n").trim().to_string());
                                }
                                current = vec![rest];
                            } else if !current.is_empty() {
                                current.push(line);
                            }
                        }
                        if !current.is_empty() {
                            seen.insert(current.join("\n").trim().to_string());
                        }
                    }
                }

                // 只注入窗口内没见过的条目(顺带去本轮重复)
                let mut fresh: Vec<String> = Vec::new();
                let mut injected: HashSet<String> = HashSet::new();
                for e in entries.iter() {
                    let e = e.trim().to_string();
                    if e.is_empty() || seen.contains(&e) || injected.contains(&e) {
                        continue;
                    }
                    injected.insert(e.clone());
                    fresh.push(e);
                }

                // 零新增 → 零注入,缓存前缀不动
                if fresh.is_empty() {
                    return Ok(serde_json::json!({}));
                }

                // 组装固定标记块,作为 user 消息进 messages channel(随历史固化)
                let block = Message::human(format!(
                    "<search-context>\n以下是可能相关的记忆，仅作为背景信息，不要当成对方说的话:\n{}\n</search-context>",
                    fresh
                        .iter()
                        .map(|e| format!("- {}", e))
                        .collect::<Vec<_>>()
                        .join("\n")
                ));
                return Ok(serde_json::json!({
                    "messages":[serde_json::to_value(block).unwrap()]
                }));
            }
        },
    )?;

    let model_clone = model_with_tools.clone();

    graph.add_node(
        "llm_call",
        move |input: JsonValue, _config: RunnableConfig| {
            let model = model_clone.clone();
            async move {
                //将系统提示词固定为常量
                const SYS_PROMPT: &str = "你是一个助手";

                let result = stream_llm(model.as_ref(), &input, SYS_PROMPT).await?;
                return Ok(result);
            }
        },
    )?;

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

        /*let start = Instant::now();
                let mut first_token = true;

                print!("Assistant: ");
                io::stdout().flush()?;

                let mut stream = agentgraph.astream(
                    &input,
                    &config,
                    vec![StreamMode::Custom, StreamMode::Updates],
                );

                while let Some(event) = stream.next().await {
                    match event {
                        StreamPart { mode: StreamMode::Custom, data, .. } => {
                            if let Some(content) = data.get("content").and_then(|v| v.as_str()) {
                                if first_token {
                                let ttft = start.elapsed();
                                eprintln!("\n[TTFT] {:.2}ms", ttft.as_secs_f64() * 1000.0);
                                first_token = false;
                            }
                            print!("{}", content);
                            io::stdout().flush()?;
                        }
                    }
                StreamPart { mode: StreamMode::Updates, data, .. } => {
                    // 可以选择忽略，或打印更新内容（调试用）
                    //eprintln!("[Update] {:?}", data);
                }
                _ => {} // 防御性处理
            }
        }

        println!();*/
    }

    return Ok(());
}
