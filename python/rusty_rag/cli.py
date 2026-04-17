"""RustyRAG CLI for local PDF question answering."""

from __future__ import annotations

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from .rag import RETRIEVAL_MODES

console = Console()
RETRIEVAL_MODE_OPTION = click.Choice(RETRIEVAL_MODES, case_sensitive=False)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """RustyRAG - chat with your local PDF documents using local retrieval."""
    load_dotenv()


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--retrieval-mode",
    type=RETRIEVAL_MODE_OPTION,
    help="Retrieval mode to prepare during ingest: bm25, hybrid, or vector.",
)
def ingest(file_path: str, retrieval_mode: str | None):
    """Ingest a PDF into the local knowledge base."""
    from .rag import ingest as do_ingest

    try:
        do_ingest(file_path, retrieval_mode=retrieval_mode)
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc


@main.command()
@click.argument("question")
@click.option(
    "--retrieval-mode",
    type=RETRIEVAL_MODE_OPTION,
    help="Query with bm25, hybrid, or vector retrieval.",
)
def query(question: str, retrieval_mode: str | None):
    """Query the knowledge base with a question."""
    from .rag import query as do_query

    try:
        response = do_query(question, retrieval_mode=retrieval_mode)
        console.print()
        console.print(Panel(response, title="Answer", border_style="green"))
        console.print()
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
