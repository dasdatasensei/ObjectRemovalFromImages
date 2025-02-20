import os
import typer
from typing import List, Optional
from pathlib import Path
import logging
from rich.console import Console
from rich.logging import RichHandler
from src.object_removal import remove_objects_from_image

# Initialize typer app and rich console
app = typer.Typer(help="🚀 AI-powered object removal tool using YOLOv8")
console = Console()


def setup_logging(verbose: bool):
    """Configure logging with rich formatting"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)]
    )


@app.command()
def remove(
    input_path: Path = typer.Argument(
        ..., help="Path to input image", exists=True, dir_okay=False, resolve_path=True
    ),
    objects: List[str] = typer.Option(
        None,
        "--remove",
        "-r",
        help="Objects to remove (can be specified multiple times)",
    ),
    output_dir: Optional[Path] = typer.Option(
        "data/output", "--output", "-o", help="Output directory for processed images"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """
    Remove specified objects from an image using AI-powered detection and inpainting.
    """
    setup_logging(verbose)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # If no objects specified, use defaults
    if not objects:
        objects = ["cup", "book"]
        logging.info(f"No objects specified, using defaults: {objects}")

    # Process image
    with console.status("[bold green]Processing image...") as status:
        try:
            result_path = remove_objects_from_image(str(input_path), objects)
            console.print(
                f"\n✨ [bold green]Success![/] Modified image saved at: {result_path}"
            )
        except Exception as e:
            console.print(f"\n❌ [bold red]Error:[/] {str(e)}")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
