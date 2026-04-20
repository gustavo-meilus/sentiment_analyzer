"""YouTube live chat extractor using yt-chat-downloader (no browser required)."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yt_chat_downloader import YouTubeChatDownloader


def extract_video_metadata(video_id: str) -> dict[str, Any]:
    """Extract video metadata via yt-dlp (bundled with yt-chat-downloader).

    @param video_id: YouTube video ID (e.g., "-rePkmiD5XU").
    @returns: Dict with title, channel, description, duration, views, thumbnail, etc.
    @throws: Exception if video info cannot be fetched.
    Time: O(1) network call. Space: O(1).
    """
    downloader = YouTubeChatDownloader()
    raw_info = downloader.get_video_info(video_id)

    return {
        "video_id": raw_info.get("id", video_id),
        "title": raw_info.get("title", ""),
        "channel": raw_info.get("channel", ""),
        "channel_id": raw_info.get("channel_id", ""),
        "description": (raw_info.get("description", "") or "")[:1000],
        "duration_seconds": raw_info.get("duration", 0),
        "view_count": raw_info.get("view_count", 0),
        "upload_date": raw_info.get("upload_date", ""),
        "thumbnail": raw_info.get("thumbnail", ""),
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }


def extract_chat_messages(
    video_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract all chat replay messages from a YouTube video.

    Uses yt-chat-downloader which fetches continuation tokens via YouTube's
    internal API — the same mechanism the chat replay player uses. No browser,
    no authentication, gets the complete chat history in one pass.

    @param video_id: YouTube video ID.
    @param output_dir: Directory for output files.
    @returns: Summary dict with total_messages, csv_path, json_path.
    Time: O(M) where M = total messages. Space: O(M).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "chat_messages.csv"
    json_path = output_dir / "chat_messages.json"

    downloader = YouTubeChatDownloader()
    raw_messages = downloader.download_chat(video_id, chat_type="live", quiet=True)

    if not raw_messages:
        print("  Warning: No chat messages found for this video.")
        return {"total_messages": 0, "csv_path": str(csv_path), "json_path": str(json_path)}

    captured_at = datetime.now(timezone.utc).isoformat()
    seen_ids: set[str] = set()
    processed: list[dict[str, str]] = []
    for m in raw_messages:
        row = _process_message(m, captured_at)
        if row is None:
            continue
        if row["message_id"] in seen_ids:
            continue
        seen_ids.add(row["message_id"])
        processed.append(row)

    _write_csv(processed, csv_path)
    _write_json(processed, json_path)

    print(f"  Extracted {len(processed)} messages")
    return {
        "total_messages": len(processed),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


def _process_message(raw: dict[str, Any], captured_at: str) -> dict[str, str] | None:
    """Transform a raw yt-chat-downloader message into the output schema.

    @param raw: Raw message dict from yt-chat-downloader.
    @param captured_at: ISO timestamp of extraction run.
    @returns: Processed message dict or None if message is empty.
    Time: O(1). Space: O(1).
    """
    comment = raw.get("comment", "")
    if not comment or not comment.strip():
        return None

    return {
        "message_id": raw.get("message_id", ""),
        "timestamp": raw.get("timestamp", ""),
        "time_in_seconds": str(float(raw.get("video_offset_ms", 0) or 0) / 1000.0),
        "author_hash": _anonymize_author(raw.get("user_display_name", "")),
        "message": comment.strip(),
        "author_type": _resolve_author_type(raw),
        "message_type": raw.get("message_type", "text_message"),
        "datetime": raw.get("datetime", ""),
        "captured_at": captured_at,
    }


def _resolve_author_type(raw: dict[str, Any]) -> str:
    """Determine the author type from yt-chat-downloader badges.

    @param raw: Raw message dict.
    @returns: "owner", "moderator", "member", or "" (regular).
    Time: O(B) where B = badge count. Space: O(1).
    """
    badges = raw.get("badges", [])
    if not badges:
        return ""
    badge_str = " ".join(str(b) for b in badges).lower()
    if "owner" in badge_str:
        return "owner"
    if "moderator" in badge_str:
        return "moderator"
    if "member" in badge_str:
        return "member"
    return ""


def _anonymize_author(author: str) -> str:
    """Hash an author name for PII anonymization.

    @param author: Raw author display name.
    @returns: SHA-256 hash truncated to 8 hex characters.
    Time: O(1). Space: O(1).
    """
    if not author:
        return "00000000"
    return hashlib.sha256(author.encode("utf-8")).hexdigest()[:8]


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a watch URL.

    @param url: YouTube URL in various formats.
    @returns: Video ID string.
    Time: O(1). Space: O(1).
    """
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url


def _write_csv(messages: list[dict[str, str]], path: Path) -> None:
    """Write messages to CSV.

    @param messages: Processed message dicts.
    @param path: Output CSV file path.
    Time: O(N). Space: O(1) (streaming write).
    """
    fieldnames = [
        "message_id", "timestamp", "time_in_seconds",
        "author_hash", "message", "author_type",
        "message_type", "datetime", "captured_at",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(messages)


def _write_json(messages: list[dict[str, str]], path: Path) -> None:
    """Write messages to JSON.

    @param messages: Processed message dicts.
    @param path: Output JSON file path.
    Time: O(N). Space: O(N).
    """
    path.write_text(json.dumps(messages, indent=2, ensure_ascii=False))


def run_extraction(
    url: str,
    output_dir: str | Path = "./output",
) -> dict[str, Any]:
    """Run the full YouTube chat extraction pipeline.

    @param url: YouTube watch URL or video ID.
    @param output_dir: Directory for all output files.
    @returns: Summary dict with total_messages, metadata, output paths.
    Time: O(M) where M = total messages.
    """
    video_id = _extract_video_id(url)
    output_path = Path(output_dir) / video_id
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "metadata.json"
    print(f"Extracting data for video: {video_id}")

    print("  Fetching video metadata...")
    metadata = extract_video_metadata(video_id)
    metadata["extraction_started_at"] = datetime.now(timezone.utc).isoformat()
    print(f"  Title: {metadata['title']}")
    print(f"  Channel: {metadata['channel']}")
    print(f"  Views: {metadata['view_count']:,}")
    print(f"  Duration: {metadata['duration_seconds']}s")

    print("  Downloading chat replay (all messages)...")
    chat_result = extract_chat_messages(video_id, output_path)

    metadata["total_messages"] = chat_result["total_messages"]
    metadata["extraction_finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"\nExtraction complete:")
    print(f"  Messages:  {chat_result['total_messages']:,}")
    print(f"  CSV:       {chat_result['csv_path']}")
    print(f"  JSON:      {chat_result['json_path']}")
    print(f"  Metadata:  {metadata_path}")

    return {
        "total_messages": chat_result["total_messages"],
        "csv_path": chat_result["csv_path"],
        "json_path": chat_result["json_path"],
        "metadata_path": str(metadata_path),
    }
