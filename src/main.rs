mod chunker;
mod db;
mod embeddings;
mod llm;
mod pdf;
mod rag;

use clap::{Parser, Subcommand};

/// RustyRAG — A local-first, privacy-focused CLI tool to chat with your PDF documents.
///
/// Powered by Ollama (local LLM) and Qdrant (vector database).
/// All data stays on your machine. Zero data leakage.
#[derive(Parser)]
#[command(name = "rusty_rag")]
#[command(version = "0.1.0")]
#[command(about = "Chat with your local PDF documents using RAG", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a PDF file into the knowledge base
    ///
    /// Extracts text from the PDF, splits it into semantic chunks,
    /// generates embeddings, and stores everything in Qdrant.
    Ingest {
        /// Path to the PDF file to ingest
        #[arg(value_name = "FILE_PATH")]
        file_path: String,
    },

    /// Query the knowledge base with a question
    ///
    /// Searches for relevant chunks in the vector database,
    /// then uses the LLM to generate an answer based on the context.
    Query {
        /// The question to ask about your documents
        #[arg(value_name = "QUESTION")]
        question: String,
    },
}

#[tokio::main]
async fn main() {
    // Load .env file if present
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Ingest { file_path } => rag::ingest_document(&file_path).await,
        Commands::Query { question } => match rag::query_document(&question).await {
            Ok(response) => {
                println!("\n{}\n", "=".repeat(60));
                println!("📝 Answer:\n");
                println!("{}", response);
                println!("\n{}", "=".repeat(60));
                Ok(())
            }
            Err(e) => Err(e),
        },
    };

    if let Err(e) = result {
        eprintln!("\n❌ Error: {:#}", e);
        std::process::exit(1);
    }
}
