#!/usr/bin/env python3
"""
Agentic RAG — production-grade CLI.

Usage examples:
    python main.py                            # interactive chat
    python main.py -q "What is RAG?"         # single-shot query
    python main.py --add-doc report.pdf       # index a document then chat
    python main.py --add-dir ./papers/        # index a folder then chat
    python main.py --list                     # show indexed sources
    python main.py --reingest                 # rebuild built-in corpus
    python main.py --no-wikipedia             # skip Wikipedia on re-ingest

Supported document formats: PDF, DOCX, TXT, MD, HTML, CSV, RST
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

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

_STEP_DISPLAY: dict[str, tuple[str, str, str]] = {
    "rewriting":              ("2",  "cyan",    "Rewriting query for retrieval"),
    "checking_details":       ("4",  "yellow",  "Checking whether retrieval is needed"),
    "selecting_source":       ("5",  "blue",    "Selecting best source"),
    "retrieving":             ("6",  "magenta", "Retrieving context"),
    "generating":             ("8",  "green",   "Generating response"),
    "checking_relevance":     ("10", "yellow",  "Evaluating response quality"),
    "complete":               ("11", "green",   "Answer accepted ✓"),
    "retry":                  ("↩",  "red",     "Insufficient — retrying"),
    "max_iterations_reached": ("!",  "red",     "Max iterations reached"),
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
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler("logs/agentic_rag.log", maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Agentic RAG — ask questions about any document",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-q", "--query",    type=str, default="", help="Run a single query then exit")
    p.add_argument("--add-doc",        type=str, default="", metavar="FILE",
                   help="Index a document (PDF/DOCX/TXT/MD/HTML/CSV) then start chat")
    p.add_argument("--add-dir",        type=str, default="", metavar="DIR",
                   help="Index all supported files in a directory then start chat")
    p.add_argument("--list",           action="store_true", help="List all indexed sources and exit")
    p.add_argument("--reingest",       action="store_true", help="Force re-ingest built-in corpus")
    p.add_argument("--no-wikipedia",   action="store_true", help="Skip Wikipedia on ingest")
    p.add_argument("-v", "--verbose",  action="store_true", help="Enable DEBUG logging")
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
                "Or export it in your shell:\n"
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
            "[dim]11-step loop  •  Claude claude-sonnet-4-6  •  ChromaDB  •  PDF / DOCX / MD / TXT / HTML[/dim]",
            border_style="blue",
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Knowledge base helpers
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
    console.print(f"[yellow]{action} built-in corpus…[/yellow] [dim](first run ~1 min)[/dim]")

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        corpus_dir = Path(__file__).parent / "agentic_rag" / "data" / "corpus"
        local_files = list(corpus_dir.glob("*.txt"))
        task = progress.add_task("Indexing corpus", total=len(local_files))

        def cb(name: str, cur: int, total: int) -> None:
            progress.update(task, description=f"[cyan]{name[:45]}[/cyan]", completed=cur)

        total_chunks = ingest(vs, use_wikipedia=use_wikipedia, progress_cb=cb)

    console.print(f"[green]✓ Indexed {total_chunks:,} chunks — {vs.count():,} total[/green]\n")
    return vs


def _add_document(vs, path_str: str) -> None:
    from agentic_rag.data.ingest import ingest_file
    from agentic_rag.loaders.dispatcher import SUPPORTED_EXTENSIONS

    path = Path(path_str)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        return

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        console.print(
            f"[red]Unsupported format:[/red] {ext}\n"
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return

    console.print(f"[bold]Indexing:[/bold] {path.name}")
    with console.status(f"[cyan]Parsing and embedding {path.name}…[/cyan]"):
        try:
            n = ingest_file(vs, path)
            console.print(f"[green]✓ Added {n} chunks from[/green] [bold]{path.name}[/bold]\n")
        except Exception as exc:
            console.print(f"[red]Failed to index {path.name}:[/red] {exc}\n")


def _add_directory(vs, dir_str: str) -> None:
    from agentic_rag.data.ingest import ingest_directory
    from agentic_rag.loaders.dispatcher import SUPPORTED_EXTENSIONS

    dir_path = Path(dir_str)
    if not dir_path.is_dir():
        console.print(f"[red]Not a directory:[/red] {dir_path}")
        return

    candidates = [
        p for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not candidates:
        console.print(f"[yellow]No supported files found in:[/yellow] {dir_path}")
        return

    console.print(f"[bold]Indexing directory:[/bold] {dir_path}  ({len(candidates)} files)")

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing files", total=len(candidates))

        def cb(name: str, cur: int, total: int) -> None:
            progress.update(task, description=f"[cyan]{name[:45]}[/cyan]", completed=cur)

        n, errors = ingest_directory(vs, dir_path, recursive=True, progress_cb=cb)

    console.print(f"[green]✓ Added {n:,} chunks from {len(candidates) - len(errors)} file(s)[/green]")
    if errors:
        console.print(f"[yellow]Skipped {len(errors)} file(s):[/yellow]")
        for e in errors:
            console.print(f"  [dim]• {e}[/dim]")
    console.print()


def _show_sources(vs) -> None:
    sources = vs.list_sources()
    if not sources:
        console.print("[yellow]No documents indexed yet.[/yellow]")
        return

    table = Table(title=f"Indexed Sources ({vs.count():,} total chunks)", show_lines=True)
    table.add_column("Title / Filename", style="bold")
    table.add_column("Type", style="cyan", justify="center")
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Source", style="dim")

    for s in sources:
        table.add_row(
            s["title"],
            s["filetype"],
            str(s["chunks"]),
            s["source"][:60] + ("…" if len(s["source"]) > 60 else ""),
        )
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

def _run_query(pipeline, query: str) -> None:
    console.print(f"\n[bold blue]Query:[/bold blue] {query}")
    console.print(Rule(style="dim"))

    state = pipeline.run(query, on_event=_on_event)

    console.print(Rule(style="dim"))

    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="dim")
    meta.add_column()
    meta.add_row("Iterations", f"{state.iteration} / {__import__('agentic_rag.config', fromlist=['config']).config.max_iterations}")
    meta.add_row("Source", state.source_used or "llm_knowledge")
    meta.add_row("Relevance score", f"{state.relevance_score}/10")
    if state.relevance_feedback:
        meta.add_row("Evaluator note", state.relevance_feedback)
    console.print(meta)
    console.print()

    border = "green" if state.is_complete else "yellow"
    title = (
        "[bold green]Final Response[/bold green]"
        if state.is_complete
        else "[bold yellow]Best Response (max iterations reached)[/bold yellow]"
    )
    console.print(Panel(state.response, title=title, border_style=border, padding=(1, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Interactive help
# ---------------------------------------------------------------------------

_HELP = """
[bold]Commands:[/bold]
  [cyan]add <file>[/cyan]       Index a document (PDF/DOCX/TXT/MD/HTML/CSV)
  [cyan]add-dir <dir>[/cyan]    Index all supported files in a folder
  [cyan]list[/cyan]             Show all indexed sources
  [cyan]reingest[/cyan]         Rebuild the built-in AI/ML corpus
  [cyan]clear[/cyan]            Clear the terminal
  [cyan]help[/cyan]             Show this message
  [cyan]exit[/cyan]             Quit

Anything else is treated as a question to ask.
"""


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

    # --- One-shot flags that don't require the pipeline ---
    if args.list:
        _show_sources(vs)
        return

    if args.add_doc:
        _add_document(vs, args.add_doc)

    if args.add_dir:
        _add_directory(vs, args.add_dir)

    # --- Single-shot query mode ---
    if args.query:
        from agentic_rag.pipeline.rag_pipeline import AgenticRAGPipeline
        _run_query(AgenticRAGPipeline(vs), args.query)
        return

    # --- Interactive loop ---
    from agentic_rag.pipeline.rag_pipeline import AgenticRAGPipeline
    pipeline = AgenticRAGPipeline(vs)

    console.print("[dim]Type a question, or type [bold]help[/bold] for commands.[/dim]\n")

    while True:
        try:
            raw = console.input("[bold green]You »[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("exit", "quit", "q", ":q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        if cmd == "help":
            console.print(_HELP)
            continue

        if cmd == "clear":
            console.clear()
            continue

        if cmd == "list":
            _show_sources(vs)
            continue

        if cmd == "reingest":
            vs = _init_knowledge_base(reingest=True, use_wikipedia=use_wikipedia)
            pipeline = AgenticRAGPipeline(vs)
            continue

        if cmd.startswith("add "):
            _add_document(vs, raw[4:].strip())
            # Rebuild pipeline so it uses the updated store
            pipeline = AgenticRAGPipeline(vs)
            continue

        if cmd.startswith("add-dir "):
            _add_directory(vs, raw[8:].strip())
            pipeline = AgenticRAGPipeline(vs)
            continue

        _run_query(pipeline, raw)


if __name__ == "__main__":
    main()
