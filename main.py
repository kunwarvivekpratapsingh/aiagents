#!/usr/bin/env python3
"""
Agentic RAG — production-grade CLI.

Usage:
    python main.py                       # interactive chat loop
    python main.py --reingest            # force re-ingest corpus
    python main.py -q "What is RAG?"    # single-shot query
    python main.py --no-wikipedia        # skip optional Wikipedia fetch
    python main.py --verbose             # enable DEBUG logging
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# Step-event display map
# ---------------------------------------------------------------------------

_STEP_DISPLAY: dict[str, tuple[str, str, str]] = {
    # event              step   colour    label
    "rewriting":         ("2",  "cyan",   "Rewriting query for retrieval"),
    "checking_details":  ("4",  "yellow", "Checking whether retrieval is needed"),
    "selecting_source":  ("5",  "blue",   "Selecting best source"),
    "retrieving":        ("6",  "magenta","Retrieving context"),
    "generating":        ("8",  "green",  "Generating response"),
    "checking_relevance":("10", "yellow", "Evaluating response quality"),
    "complete":          ("11", "green",  "Answer accepted ✓"),
    "retry":             ("↩",  "red",    "Insufficient answer — retrying"),
    "max_iterations_reached": ("!", "red","Max iterations reached"),
}


def _on_event(event: str, state) -> None:
    step, colour, label = _STEP_DISPLAY.get(event, ("?", "white", event))
    extra = ""
    if event == "selecting_source" and state.source_used:
        extra = f"  [dim]→ {state.source_used}[/dim]"
    if event == "checking_relevance" and state.relevance_score:
        extra = f"  [dim]score {state.relevance_score}/10[/dim]"
    if event == "retry":
        extra = f"  [dim](iteration {state.iteration})[/dim]"
    console.print(f"  [{colour}]●[/{colour}] [bold]Step {step}[/bold]  {label}{extra}")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")

    # Always write DEBUG to a rotating log file regardless of console level
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    fh = RotatingFileHandler(
        f"{log_dir}/agentic_rag.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agentic RAG — production-grade 11-step retrieval loop"
    )
    p.add_argument("--reingest", action="store_true", help="Force re-ingest the corpus")
    p.add_argument("--no-wikipedia", action="store_true", help="Skip Wikipedia ingestion")
    p.add_argument("-q", "--query", type=str, default="", help="Run a single query then exit")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    return p


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def _check_api_key() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print(
            Panel(
                "[bold red]ANTHROPIC_API_KEY is not set.[/bold red]\n\n"
                "Add it to a [bold].env[/bold] file:\n"
                "  ANTHROPIC_API_KEY=sk-ant-...\n\n"
                "Or export it before running:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...",
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
            "[dim]11-step agentic loop  •  Claude claude-sonnet-4-6  •  ChromaDB  •  15-topic local corpus[/dim]",
            border_style="blue",
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Knowledge base init
# ---------------------------------------------------------------------------

def _init_knowledge_base(reingest: bool, use_wikipedia: bool):
    from agentic_rag.sources.vector_store import VectorStore
    from agentic_rag.data.ingest import ingest

    console.print("[bold]Initializing knowledge base…[/bold]")
    vs = VectorStore()
    doc_count = vs.count()

    if doc_count > 0 and not reingest:
        console.print(f"[green]✓ Knowledge base ready:[/green] {doc_count:,} chunks indexed\n")
        return vs

    action = "Re-ingesting" if doc_count > 0 else "Building"
    console.print(f"[yellow]{action} corpus…[/yellow] [dim](first run may take 1–2 min)[/dim]")

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        # Local corpus
        local_files = list((__import__("pathlib").Path(__file__).parent / "agentic_rag" / "data" / "corpus").glob("*.txt"))
        local_task = progress.add_task("Local corpus", total=len(local_files))
        wiki_task = None
        if use_wikipedia:
            from agentic_rag.data.ingest import WIKIPEDIA_TOPICS
            wiki_task = progress.add_task("Wikipedia (optional)", total=len(WIKIPEDIA_TOPICS))

        _phase: list[str] = ["local"]

        def cb(topic: str, current: int, total: int) -> None:
            if _phase[0] == "local":
                progress.update(local_task, description=f"[cyan]{topic[:40]}[/cyan]", completed=current)
                if current >= total and wiki_task is not None:
                    _phase[0] = "wiki"
            else:
                if wiki_task is not None:
                    progress.update(wiki_task, description=f"[dim]{topic[:40]}[/dim]", completed=current)

        total_chunks = ingest(vs, use_wikipedia=use_wikipedia, progress_cb=cb)

    console.print(f"[green]✓ Indexed {total_chunks:,} chunks — {vs.count():,} total in store[/green]\n")
    return vs


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

def _run_query(pipeline, query: str) -> None:
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(Rule(style="dim"))

    state = pipeline.run(query, on_event=_on_event)

    console.print(Rule(style="dim"))

    # Metadata summary
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim")
    meta.add_column()
    meta.add_row("Iterations used", f"{state.iteration} / {__import__('agentic_rag.config', fromlist=['config']).config.max_iterations}")
    meta.add_row("Source used", state.source_used or "llm_knowledge")
    meta.add_row("Final relevance score", f"{state.relevance_score}/10")
    if state.relevance_feedback:
        meta.add_row("Evaluator feedback", state.relevance_feedback)
    console.print(meta)
    console.print()

    border = "green" if state.is_complete else "yellow"
    title = "[bold green]Final Response[/bold green]" if state.is_complete else "[bold yellow]Best Response (max iterations)[/bold yellow]"
    console.print(Panel(state.response, title=title, border_style=border, padding=(1, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    _setup_logging(args.verbose)
    _check_api_key()
    _print_banner()

    use_wikipedia = not args.no_wikipedia
    vs = _init_knowledge_base(args.reingest, use_wikipedia)

    from agentic_rag.pipeline.rag_pipeline import AgenticRAGPipeline

    pipeline = AgenticRAGPipeline(vs)

    # Single-shot mode
    if args.query:
        _run_query(pipeline, args.query)
        return

    # Interactive loop
    console.print(
        "[dim]Ask anything. Commands: [bold]exit[/bold] | [bold]reingest[/bold] | [bold]clear[/bold][/dim]\n"
    )

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

        if query.lower() == "clear":
            console.clear()
            continue

        if query.lower() == "reingest":
            vs = _init_knowledge_base(reingest=True, use_wikipedia=use_wikipedia)
            pipeline = AgenticRAGPipeline(vs)
            continue

        _run_query(pipeline, query)


if __name__ == "__main__":
    main()
