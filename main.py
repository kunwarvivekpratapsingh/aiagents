#!/usr/bin/env python3
"""
Agentic RAG — production-grade CLI entry point.

Usage:
    python main.py              # interactive chat loop
    python main.py --reingest   # force re-ingestion of Wikipedia corpus
    python main.py --query "What is RAG?"  # single-shot query
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# Step-event display
# ---------------------------------------------------------------------------

_STEP_ICONS: dict[str, tuple[str, str]] = {
    "rewriting":          ("2", "cyan",    "Rewriting query for retrieval"),
    "checking_details":   ("4", "yellow",  "Checking whether retrieval is needed"),
    "selecting_source":   ("5", "blue",    "Selecting best source"),
    "retrieving":         ("6", "magenta", "Retrieving context"),
    "generating":         ("8", "green",   "Generating response"),
    "checking_relevance": ("10","yellow",  "Evaluating response quality"),
    "complete":           ("11","green",   "Answer accepted"),
    "retry":              ("↩", "red",     "Insufficient — retrying"),
}


def _on_event(event: str, state) -> None:
    step, color, label = _STEP_ICONS.get(event, ("?", "white", event))
    extra = ""
    if event == "selecting_source" and state.source_used:
        extra = f" [dim]→ {state.source_used}[/dim]"
    if event == "checking_relevance" and state.relevance_score:
        extra = f" [dim]score: {state.relevance_score}/10[/dim]"
    if event == "retry":
        extra = f" [dim](iter {state.iteration})[/dim]"
    console.print(
        f"  [{color}]●[/{color}] [bold]Step {step}[/bold]  {label}{extra}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agentic RAG System")
    p.add_argument("--reingest", action="store_true", help="Force re-ingest Wikipedia corpus")
    p.add_argument("--query", "-q", type=str, default="", help="Run a single query then exit")
    p.add_argument("--topics-file", type=str, default="", help="Path to a .txt file (one topic per line) for ingestion")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    return p


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _check_api_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print(
            Panel(
                "[bold red]ANTHROPIC_API_KEY is not set.[/bold red]\n\n"
                "Create a [bold].env[/bold] file with:\n"
                "  ANTHROPIC_API_KEY=sk-ant-...\n\n"
                "Or export it in your shell before running.",
                title="Missing API Key",
                border_style="red",
            )
        )
        sys.exit(1)


def _print_banner() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]Agentic RAG System[/bold blue]\n"
            "[dim]11-step agentic retrieval loop  •  Claude Sonnet  •  ChromaDB  •  Wikipedia corpus[/dim]",
            border_style="blue",
        )
    )
    console.print()


def _init_knowledge_base(reingest: bool, topics_file: str):
    from agentic_rag.sources.vector_store import VectorStore
    from agentic_rag.data.ingest import ingest_wikipedia, WIKIPEDIA_TOPICS

    console.print("[bold]Initializing knowledge base…[/bold]")
    vs = VectorStore()
    doc_count = vs.count()

    if doc_count > 0 and not reingest:
        console.print(f"[green]✓ Knowledge base ready:[/green] {doc_count:,} chunks indexed\n")
        return vs

    topics = WIKIPEDIA_TOPICS
    if topics_file:
        with open(topics_file) as f:
            topics = [line.strip() for line in f if line.strip()]

    console.print(
        f"[yellow]Ingesting {len(topics)} Wikipedia articles…[/yellow] "
        "[dim](first-time setup, ~1–3 min)[/dim]"
    )

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

    ingested_count: list[int] = [0]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching articles", total=len(topics))

        def cb(topic: str, current: int, total: int) -> None:
            progress.update(task, description=f"[cyan]{topic[:45]}[/cyan]", completed=current)

        ingested_count[0] = ingest_wikipedia(vs, topics=topics, progress_cb=cb)

    console.print(f"[green]✓ Ingested {ingested_count[0]:,} chunks from {len(topics)} articles[/green]\n")
    return vs


def _run_query(pipeline, query: str) -> None:
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(Rule(style="dim"))

    state = pipeline.run(query, on_event=_on_event)

    console.print(Rule(style="dim"))

    # Metadata summary table
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim")
    meta.add_column()
    meta.add_row("Iterations", str(state.iteration))
    meta.add_row("Source used", state.source_used or "llm_knowledge")
    meta.add_row("Relevance score", f"{state.relevance_score}/10")
    console.print(meta)
    console.print()

    console.print(
        Panel(
            state.response,
            title="[bold green]Final Response[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()


def main() -> None:
    args = build_parser().parse_args()
    _setup_logging(args.verbose)
    _check_api_key()
    _print_banner()

    vs = _init_knowledge_base(args.reingest, args.topics_file)

    from agentic_rag.pipeline.rag_pipeline import AgenticRAGPipeline

    pipeline = AgenticRAGPipeline(vs)

    # Single-shot mode
    if args.query:
        _run_query(pipeline, args.query)
        return

    # Interactive loop
    console.print("[dim]Type your question and press Enter. Commands: [bold]exit[/bold] | [bold]reingest[/bold][/dim]\n")

    while True:
        try:
            query = console.input("[bold green]You »[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q", ":q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if query.lower() == "reingest":
            vs = _init_knowledge_base(reingest=True, topics_file="")
            from agentic_rag.pipeline.rag_pipeline import AgenticRAGPipeline
            pipeline = AgenticRAGPipeline(vs)
            continue

        _run_query(pipeline, query)


if __name__ == "__main__":
    main()
