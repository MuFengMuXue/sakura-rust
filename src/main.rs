use std::sync::Arc;
use std::io::{self, Write};

use serde_json::Value as JsonValue;
use langgraph::checkpoint::InMemorySaver;
use dotenvy::dotenv;
use langgraph::prelude::*;
use langgraph::langgraph_state;
use langgraph::prebuilt::{BaseChatModel, Message, print_stream, response_text, stream_llm};
use langgraph::providers::openai::{OpenAIModelConfig,OpenAIModel};


//定义state
#[langgraph_state]
#[derive(Debug)]
struct GraphState{
    #[channel(messages)]
    messages: Vec<Message>,
    search_context: String,
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
    let (api_key, api_base, model_name) = load_openai_config();
    let model = OpenAIModel::new(OpenAIModelConfig {
        model: model_name,
        api_key,
        api_base,
        ..Default::default()
    });
    let model_with_tools: Arc<dyn BaseChatModel> = Arc::new(model);

    let channels = GraphState::create_channels();
    let mut graph = StateGraph::new(channels);


    graph.add_node("search_memories", move |input: JsonValue, _config: RunnableConfig| {
        async move {
        // 提取最后一条用户消息
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

            let user_id = "01";
            let client = reqwest::Client::new();

            let payload = serde_json::json!({
                "query": last_user_msg,
                "user_id": user_id,
            });

        // 发送请求，并将 reqwest::Error 转换为 RunnableError
            let response = client
                .post("http://127.0.0.1:8004/search")
                .json(&payload)
                .send()
                .await
                .map_err(|e| RunnableError::Other(Box::new(e)))?;

            let context = if response.status().is_success() {
            // 解析 JSON，同样需要转换错误
                let data: serde_json::Value = response
                    .json()
                    .await
                    .map_err(|e| RunnableError::Other(Box::new(e)))?;
                data.get("memories")
                    .and_then(|v| v.as_array())
                    .map(|arr|{
                        arr.iter()
                        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n")
                    })
                    .unwrap_or_else(|| String::new())
            } else {
                eprintln!("请求失败，状态码: {}", response.status());
                String::new()
            };

            return Ok(serde_json::json!({"search_context": context}));
        }
    })?;

    graph.add_node("llm_call", move |input: JsonValue, _config: RunnableConfig| {
        let model = model_with_tools.clone();
        async move {
            // stream_llm 会通过 get_stream_writer 发送 token 到 Custom 流
            let context = input.get("search_context").and_then(|c| c.as_str()).unwrap_or("");
            let mut sys_prompt = "你是一个助手".to_string();
            if !context.is_empty(){
                sys_prompt.push_str(&format!("\n以下是**可能**有关的记忆，如果不相关可以不使用：\n{}", context));
            }
            let result = stream_llm(
                model.as_ref(),
                &input,
                &sys_prompt,
            ).await?;
            let text = response_text(&result);
            let ai_message = serde_json::json!({
                "type": "ai",
                "content": text
            });
            // 返回更新，追加到 messages
            return Ok(serde_json::json!({
                "messages": [ai_message]
            }));
        }
    })?;

    graph.add_edge(START, "search_memories")?;
    graph.add_edge("search_memories", "llm_call")?;
    graph.add_edge("llm_call", END)?;
    let checkpointer = Arc::new(InMemorySaver::new());
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