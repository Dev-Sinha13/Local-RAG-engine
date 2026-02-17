"""Ollama LLM completion with context-aware prompting."""

import os
import ollama


def ask(question: str, context: str = "", model: str | None = None) -> str:
    """Send a prompt to the local LLM with optional RAG context.

    If context is provided, the model is instructed to only answer
    based on the given context. Otherwise, it acts as a general assistant.
    """
    model = model or os.getenv("COMPLETION_MODEL", "llama3.2")

    if context:
        system = (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the following context.\n"
            'If the answer is not in the context, say "I don\'t have enough '
            'information to answer that."\n\n'
            f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---"
        )
    else:
        system = "You are a helpful assistant."

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )

    return response["message"]["content"]
