"""CLI entry point for the YouTube live chat extractor."""

import argparse
import sys

from extractor.chat_extractor import run_extraction


def main() -> None:
    """Parse arguments and run the YouTube chat extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract chat messages from YouTube live stream replays using chat-downloader.",
    )
    parser.add_argument("--url", required=True, help="YouTube watch URL with live chat replay.")
    parser.add_argument("--output-dir", default="./output", help="Output directory (default: ./output).")

    args = parser.parse_args()

    try:
        run_extraction(url=args.url, output_dir=args.output_dir)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExtraction interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
