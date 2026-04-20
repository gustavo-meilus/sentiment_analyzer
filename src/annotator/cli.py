"""CLI entry point for the sentiment annotation TUI tool."""

import argparse
import csv
import sys
from pathlib import Path

from annotator.tui import run_tui


def load_sample(path: Path) -> list[dict]:
    """Read the sampled messages CSV into a list of message dicts.

    O(n) where n is the number of rows in the CSV.

    @param path: Path to the input CSV file (must have row_id, message, timestamp, time_in_seconds).
    @returns: List of dicts, each representing one chat message.
    @throws FileNotFoundError: If the input path does not exist.
    @throws ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    required = {"row_id", "message", "timestamp", "time_in_seconds"}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {path}")

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        # video_id is optional (present in multi-video exports)

        return list(reader)


def load_progress(path: Path) -> dict[int, dict]:
    """Read existing annotation output for resume, keyed by row_id.

    O(n) where n is the number of rows in the progress CSV.

    @param path: Path to the existing output CSV.
    @returns: Dict mapping row_id (int) to annotation dict with keys row_id, label, aspect.
    """
    if not path.exists():
        return {}

    result: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row["row_id"])
            result[rid] = {"row_id": rid, "label": row["label"], "aspect": row["aspect"]}

    return result


def save_annotations(annotations: list[dict], path: Path) -> None:
    """Write annotation dicts to a CSV file with columns row_id, label, aspect.

    O(n) where n is the number of annotations.

    @param annotations: List of annotation dicts (each has row_id, label, aspect).
    @param path: Output file path. Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "label", "aspect"])
        writer.writeheader()
        for ann in annotations:
            writer.writerow({"row_id": ann["row_id"], "label": ann["label"], "aspect": ann["aspect"]})


def main() -> None:
    """Parse arguments, run the annotation TUI, and write output CSV.

    @throws SystemExit: On argument errors or fatal exceptions.
    """
    parser = argparse.ArgumentParser(
        description="Curses TUI for fast human labeling of YouTube chat messages.",
    )
    parser.add_argument("--input", required=True, help="Path to sampled messages CSV.")
    parser.add_argument(
        "--output",
        default="./output/ground_truth_human.csv",
        help="Path for annotated output CSV (default: ./output/ground_truth_human.csv).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from last annotated row.")
    parser.add_argument(
        "--context",
        default=None,
        help="Path to full chat CSV (e.g. chat_messages.csv) for the context pane.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        messages = load_sample(input_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    existing: dict[int, dict] = {}
    if args.resume:
        existing = load_progress(output_path)
        if existing:
            print(f"Resuming: {len(existing)} annotations loaded from {output_path}")

    context_messages: list[dict] = []
    if args.context:
        context_path = Path(args.context)
        if not context_path.exists():
            print(f"Warning: context file not found: {context_path}", file=sys.stderr)
        else:
            with open(context_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                context_messages = sorted(
                    list(reader),
                    key=lambda m: float(m.get("time_in_seconds") or 0),
                )
            print(f"Context loaded: {len(context_messages):,} messages from {context_path.name}")

    def on_save(annotations: list[dict]) -> None:
        save_annotations(annotations, output_path)

    try:
        annotations = run_tui(messages, existing, save_callback=on_save, context_messages=context_messages)
    except KeyboardInterrupt:
        print("\nAnnotation interrupted by user.")
        sys.exit(0)

    if annotations:
        save_annotations(annotations, output_path)
        print(f"Saved {len(annotations)} annotations to {output_path}")


if __name__ == "__main__":
    main()
