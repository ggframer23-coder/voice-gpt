from __future__ import annotations

from typing import Optional

import typer

from .journal import search
from .settings import load_settings

app = typer.Typer(add_completion=False)


def _print_query_results(results: list[dict]) -> None:
    if not results:
        print("No results")
        return
    for idx, item in enumerate(results, start=1):
        print(f"[{idx}] score={item['score']:.3f} entry={item['entry_id']} chunk={item['chunk_id']}")
        if item.get("recorded_at"):
            print(f"recorded_at={item['recorded_at']}")
        print(item["chunk_text"])
        print()


@app.command()
def query(
    q: Optional[str] = typer.Argument(None, help="Query text."),
    k: int = typer.Option(5, help="Top results to return."),
    recorded_from: Optional[str] = typer.Option(None, help="Filter by recorded_at >= (ISO 8601)."),
    recorded_to: Optional[str] = typer.Option(None, help="Filter by recorded_at <= (ISO 8601)."),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Prompt for queries in a loop."),
) -> None:
    """Search the journal using FAISS."""
    settings = load_settings()

    def run_query(text: str) -> None:
        results = search(settings, query=text, k=k, recorded_from=recorded_from, recorded_to=recorded_to)
        _print_query_results(results)

    if interactive:
        print("Interactive mode. Enter a query (type 'quit' to exit).")
        while True:
            try:
                text = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in {"quit", "exit"}:
                break
            if text:
                run_query(text)
        return

    if not q:
        raise typer.Exit("Query text is required unless --interactive is set.")
    run_query(q)
