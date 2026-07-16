"""CLI entrypoint for the fusion engine.

Examples
--------
::

    # Any subset of the four inputs -- pass only what you have
    python scripts/analyze.py --video clip.mp4 --text "the transcript"
    python scripts/analyze.py --image photo.jpg --audio clip.wav
    python scripts/analyze.py --video clip.mp4 --json-out result.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import typer

from fusion_engine import FusionEngine

app = typer.Typer(add_completion=False, help="Run the multi-modal fusion engine on one submission.")


@app.command()
def main(
    image: Path | None = typer.Option(None, help="Path to an image file."),
    audio: Path | None = typer.Option(None, help="Path to an audio file."),
    text: str | None = typer.Option(None, help="Text content to analyse."),
    video: Path | None = typer.Option(None, help="Path to a video file."),
    json_out: Path | None = typer.Option(None, help="Optional path to write the full JSON result."),
) -> None:
    """Run whichever modalities have an input and print the fused verdict."""
    if not any([image, audio, text, video]):
        raise typer.BadParameter("Provide at least one of --image/--audio/--text/--video.")

    engine = FusionEngine()
    if engine.load_errors:
        typer.echo("Adapter load warnings (these modalities will be skipped):", err=True)
        for modality, err in engine.load_errors.items():
            typer.echo(f"  {modality}: {err}", err=True)

    result = engine.analyze(image_path=image, audio_path=audio, text=text, video_path=video)

    typer.echo("")
    typer.echo("--- Fusion result ---")
    typer.echo(f"Verdict           : {result.verdict}")
    typer.echo(f"Fused score       : {result.fused_score:.4f}")
    typer.echo(f"Fused confidence  : {result.fused_confidence:.4f}")
    typer.echo(f"Weights used      : {result.weights_used}")
    if result.cross_modal_flags:
        typer.echo("Cross-modal flags :")
        for flag in result.cross_modal_flags:
            typer.echo(f"  - {flag}")
    typer.echo(f"Explanation       : {result.explanation}")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        typer.echo(f"\nWrote full result to {json_out}")


if __name__ == "__main__":
    app()
