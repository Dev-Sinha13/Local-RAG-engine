use anyhow::Result;
use rig::client::{CompletionClient, Nothing};
use rig::completion::Prompt;
use rig::providers::ollama;

/// Creates an Ollama client connected to the local instance.
pub fn create_client() -> Result<ollama::Client<reqwest::Client>> {
    let client: ollama::Client<reqwest::Client> = ollama::Client::new(Nothing)?;
    Ok(client)
}

/// Sends a prompt to the LLM with optional context and returns the response.
///
/// If `context` is non-empty, it is injected as a system preamble instructing
/// the model to only answer based on the provided context.
pub async fn ask_ai(question: &str, context: &str) -> Result<String> {
    let client = create_client()?;

    let model_name = std::env::var("COMPLETION_MODEL").unwrap_or_else(|_| "llama3.2".to_string());

    let preamble = if context.is_empty() {
        "You are a helpful assistant.".to_string()
    } else {
        format!(
            "You are a helpful assistant. Answer the user's question using ONLY the following context.\n\
             If the answer is not in the context, say \"I don't have enough information to answer that.\"\n\n\
             --- CONTEXT ---\n{}\n--- END CONTEXT ---",
            context
        )
    };

    let agent = client
        .agent(&model_name)
        .preamble(&preamble)
        .build();

    let response = agent.prompt(question).await?;
    Ok(response)
}
