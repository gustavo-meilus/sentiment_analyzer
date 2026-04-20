"""Curses TUI render loop and state management for sentiment annotation."""

import bisect
import curses
from dataclasses import dataclass, replace
from typing import Literal


SENTIMENT_KEYS: dict[int, str] = {
    ord("p"): "Positive",
    ord("P"): "Positive",
    ord("n"): "Negative",
    ord("N"): "Negative",
    ord("u"): "Neutral",
    ord("U"): "Neutral",
}

ASPECT_KEYS: dict[int, str] = {
    ord("c"): "content",
    ord("C"): "content",
    ord("r"): "presenter",
    ord("R"): "presenter",
    ord("m"): "community",
    ord("M"): "community",
    ord("a"): "audio_video",
    ord("A"): "audio_video",
    ord("t"): "meta",
    ord("T"): "meta",
    ord("x"): "unknown",
    ord("X"): "unknown",
}

CONTEXT_WINDOW: int = 4    # messages before/after current to show in context pane
AUTO_SAVE_INTERVAL: int = 25


@dataclass(frozen=True)
class AnnotationState:
    """Immutable state for the annotation TUI.

    @param index: Current message index in the messages list.
    @param step: Which annotation dimension is active.
    @param pending_label: Sentiment label waiting for aspect selection.
    @param annotations: Completed annotations as an immutable tuple.
    @param should_quit: Whether the user requested quit.
    @param should_save: Whether a save was requested.
    @param unsaved_count: Number of annotations since last save.
    @param ctx_offset: Scroll offset applied to the context window.
    """

    index: int
    step: Literal["sentiment", "aspect"]
    pending_label: str | None
    annotations: tuple[dict, ...]
    should_quit: bool
    should_save: bool
    unsaved_count: int
    ctx_offset: int = 0


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _find_center(times: list[float], target: float) -> int:
    """Binary search: index of the message whose time is closest to target.

    O(log n).
    """
    idx = bisect.bisect_left(times, target)
    if idx >= len(times):
        return len(times) - 1
    if idx > 0 and abs(times[idx - 1] - target) < abs(times[idx] - target):
        return idx - 1
    return idx


def _get_context_slice(
    context_messages: list[dict],
    current_time: float,
    window: int,
    offset: int,
) -> tuple[list[dict], int]:
    """Return a window of messages around current_time with a scroll offset.

    O(log n) for the binary search, O(window) for slicing.

    @param context_messages: Full stream sorted by time_in_seconds.
    @param current_time: time_in_seconds of the message being annotated.
    @param window: Number of messages before/after to include.
    @param offset: Scroll offset (negative = earlier, positive = later).
    @returns: (slice, idx_of_current_in_slice) — idx is -1 if scrolled out of view.
    """
    if not context_messages:
        return [], -1

    times = [float(m.get("time_in_seconds") or 0) for m in context_messages]
    real_center = _find_center(times, current_time)

    # Apply scroll offset, clamped so the window stays in bounds
    display_center = real_center + offset
    display_center = max(window, min(len(context_messages) - 1 - window, display_center))

    start = max(0, display_center - window)
    end = min(len(context_messages), display_center + window + 1)
    slc = context_messages[start:end]

    current_in_slice = real_center - start
    if not (0 <= current_in_slice < len(slc)):
        current_in_slice = -1

    return slc, current_in_slice


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def _handle_key(
    key: int,
    state: AnnotationState,
    total: int,
    has_context: bool,
) -> AnnotationState:
    """Pure function: map a key press to a new annotation state.

    O(1).

    @param key: Curses key code.
    @param state: Current immutable state.
    @param total: Total number of messages to annotate.
    @param has_context: Whether a context stream is loaded (enables ↑/↓).
    @returns: New AnnotationState.
    """
    if key == ord("q") or key == ord("Q"):
        return replace(state, should_quit=True, should_save=True)

    if key == ord("s") or key == ord("S"):
        return replace(state, should_save=True)

    # Context pane scrolling
    if key == curses.KEY_UP:
        return replace(state, ctx_offset=state.ctx_offset - 1)
    if key == curses.KEY_DOWN:
        return replace(state, ctx_offset=state.ctx_offset + 1)

    # Navigate back
    if key in (curses.KEY_LEFT, curses.KEY_BACKSPACE, 127, 8):
        if state.index <= 0:
            return state
        return replace(
            state,
            index=state.index - 1,
            step="sentiment",
            pending_label=None,
            ctx_offset=0,
        )

    # Annotation flow: sentiment → aspect
    if state.step == "sentiment":
        if key not in SENTIMENT_KEYS:
            return state
        return replace(state, step="aspect", pending_label=SENTIMENT_KEYS[key])

    if key not in ASPECT_KEYS:
        return state

    annotation = {
        "row_id": None,
        "label": state.pending_label,
        "aspect": ASPECT_KEYS[key],
    }
    new_annotations = state.annotations + (annotation,)
    new_index = min(state.index + 1, total)
    new_unsaved = state.unsaved_count + 1
    needs_save = new_unsaved >= AUTO_SAVE_INTERVAL

    return AnnotationState(
        index=new_index,
        step="sentiment",
        pending_label=None,
        annotations=new_annotations,
        should_quit=new_index >= total,
        should_save=needs_save,
        unsaved_count=0 if needs_save else new_unsaved,
        ctx_offset=0,
    )


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_header(win: "curses.window", index: int, total: int) -> int:
    """Render the title bar with progress indicator. Returns rows used."""
    height, width = win.getmaxyx()
    if width < 20 or height < 1:
        return 0

    pct = (index / total * 100) if total > 0 else 0
    bar_max = max(width - 50, 10)
    filled = int(bar_max * index / total) if total > 0 else 0
    bar = "\u2588" * filled + "\u2591" * (bar_max - filled)
    header = f"  SENTIMENT ANNOTATOR   [{index}/{total}]  {pct:5.1f}%  {bar}"

    try:
        win.attron(curses.A_REVERSE | curses.A_BOLD)
        win.addstr(0, 0, header[:width - 1].ljust(width - 1))
        win.attroff(curses.A_REVERSE | curses.A_BOLD)
    except curses.error:
        pass
    return 1


def _draw_context(
    win: "curses.window",
    ctx_slice: list[dict],
    current_idx: int,
    start_row: int,
    max_rows: int,
    width: int,
    show_scroll: bool = True,
) -> int:
    """Render the context message pane. Returns rows used.

    O(window) — one line per context message.

    @param ctx_slice: Messages to display.
    @param current_idx: Index in ctx_slice of the message being annotated (highlighted).
    @param start_row: First terminal row to write to.
    @param max_rows: Maximum rows available for this pane (including title).
    @param has_scroll: Whether to show the ↑/↓ scroll hint in the title.
    """
    if not ctx_slice or max_rows <= 0:
        return 0

    scroll_hint = " [↑/↓ scroll]" if show_scroll else ""
    title = f"  \u2500\u2500 Context{scroll_hint} "
    divider = title + "\u2500" * max(0, width - len(title) - 1)

    try:
        win.addstr(start_row, 0, divider[:width - 1], curses.A_DIM)
    except curses.error:
        pass
    rows_used = 1

    for i, msg in enumerate(ctx_slice):
        if rows_used >= max_rows:
            break
        ts = msg.get("timestamp", "?")
        text = msg.get("message", "")
        max_text = width - len(ts) - 7
        truncated = (text[:max_text - 1] + "\u2026") if len(text) > max_text else text
        line = f"  [{ts}] {truncated}"
        row = start_row + rows_used
        try:
            if i == current_idx:
                marker = "> " + line[2:]
                win.addstr(row, 0, marker[:width - 1], curses.A_BOLD)
            else:
                win.addstr(row, 0, line[:width - 1], curses.A_DIM)
        except curses.error:
            pass
        rows_used += 1

    return rows_used


def _draw_message(win: "curses.window", message: dict, start_row: int, width: int) -> int:
    """Render the full current message block. Returns rows used."""
    row_id = message.get("row_id", "?")
    timestamp = message.get("timestamp", "?")
    time_s = message.get("time_in_seconds", "?")
    text = message.get("message", "")

    meta = f"  row_id: {row_id}    timestamp: {timestamp}    time: {time_s}s"
    msg = f'  "{text}"'

    rows_used = 0
    try:
        win.addstr(start_row, 0, meta[:width - 1])
        rows_used += 1
        win.addstr(start_row + 1, 0, msg[:width - 1], curses.A_BOLD)
        rows_used += 1
    except curses.error:
        pass
    return rows_used


def _draw_step_indicator(win: "curses.window", step: str, row: int, width: int) -> int:
    """Draw the 'Select SENTIMENT / ASPECT' prompt. Returns rows used."""
    indicator = f"  >> Select {step.upper()} <<"
    try:
        win.addstr(row, 0, indicator[:width - 1], curses.A_BLINK | curses.A_BOLD)
    except curses.error:
        pass
    return 1


def _draw_controls(
    win: "curses.window",
    step: str,
    last_annotation: dict | None,
    start_row: int,
    width: int,
) -> None:
    """Render key binding grid and last-annotation reference. O(1)."""
    height = win.getmaxyx()[0]
    row = start_row
    divider = "\u2500" * (width - 1)

    s_items = ["[P] Positive", "[N] Negative", "[U] Neutral"]
    a_items = [
        "[C] content",
        "[R] presenter",
        "[M] community",
        "[A] audio_video",
        "[T] meta",
        "[X] unknown",
    ]

    try:
        if row < height - 1:
            win.addstr(row, 0, divider[:width - 1])
            row += 1

        if row < height - 1:
            mid = width // 2
            win.addstr(row, 0, "  SENTIMENT"[:mid - 1])
            win.addstr(row, mid, "ASPECT"[:width - mid - 1])
            row += 1

        for i in range(max(len(s_items), len(a_items))):
            if row >= height - 3:
                break
            mid = width // 2
            if i < len(s_items):
                attr = curses.A_BOLD if step == "sentiment" else curses.A_DIM
                win.addstr(row, 0, f"  {s_items[i]}"[:mid - 1], attr)
            if i < len(a_items):
                attr = curses.A_BOLD if step == "aspect" else curses.A_DIM
                win.addstr(row, mid, a_items[i][:width - mid - 1], attr)
            row += 1

        if row < height - 2:
            win.addstr(row, 0, divider[:width - 1])
            row += 1

        if row < height - 1:
            nav = "  [↑/↓] scroll context   [←] back   [S] save   [Q] save & quit"
            win.addstr(row, 0, nav[:width - 1])
            row += 1

        if row < height and last_annotation is not None:
            last_msg = (
                f"  Last: row {last_annotation['row_id']} "
                f"\u2192 {last_annotation['label']} / {last_annotation['aspect']}"
            )
            win.addstr(row, 0, last_msg[:width - 1], curses.A_DIM)

    except curses.error:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_tui(
    messages: list[dict],
    existing: dict[int, dict],
    save_callback=None,
    context_messages: list[dict] | None = None,
) -> list[dict]:
    """Curses main loop for annotating messages. Returns completed annotations.

    O(n * k) where n is messages and k is keypresses per message (ideally 2).

    @param messages: GT sample messages to annotate.
    @param existing: Already-annotated row_ids for resume support.
    @param save_callback: Optional callable invoked on save events.
    @param context_messages: Full stream sorted by time for context pane (optional).
    @returns: List of annotation dicts with keys row_id, label, aspect.
    """
    # Sort by time so the context pane and annotation flow are in stream order
    messages = sorted(messages, key=lambda m: float(m.get("time_in_seconds") or 0))

    pre_annotations = []
    start_index = 0
    for i, msg in enumerate(messages):
        rid = int(msg["row_id"])
        if rid in existing:
            pre_annotations.append(existing[rid])
            start_index = i + 1
        else:
            break

    return curses.wrapper(
        _curses_main,
        messages,
        pre_annotations,
        start_index,
        save_callback,
        context_messages or [],
    )


def _curses_main(
    stdscr,
    messages: list[dict],
    pre_annotations: list[dict],
    start_index: int,
    save_callback,
    context_messages: list[dict],
) -> list[dict]:
    """Inner curses loop called by curses.wrapper."""
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    total = len(messages)
    has_context = bool(context_messages)

    state = AnnotationState(
        index=start_index,
        step="sentiment",
        pending_label=None,
        annotations=tuple(pre_annotations),
        should_quit=False,
        should_save=False,
        unsaved_count=0,
        ctx_offset=0,
    )

    while True:
        height, width = stdscr.getmaxyx()
        stdscr.erase()

        if height < 12 or width < 40:
            try:
                stdscr.addstr(0, 0, "Terminal too small. Resize to at least 40x12.")
            except curses.error:
                pass
            stdscr.refresh()
            key = stdscr.getch()
            if key in (ord("q"), ord("Q")):
                return list(state.annotations)
            continue

        if state.should_quit or state.index >= total:
            if state.should_save and save_callback is not None:
                save_callback(list(state.annotations))
            return list(state.annotations)

        current_msg = messages[state.index]
        last_ann = state.annotations[-1] if state.annotations else None
        current_time = float(current_msg.get("time_in_seconds") or 0)

        row = _draw_header(stdscr, state.index, total)

        # --- Context pane ---
        # Allocate up to CONTEXT_WINDOW*2+1 message rows + 1 title row, but cap at
        # half the terminal so controls always fit.
        ctx_budget = min(CONTEXT_WINDOW * 2 + 2, height // 2)

        if has_context:
            ctx_slice, current_in_slice = _get_context_slice(
                context_messages, current_time, CONTEXT_WINDOW, state.ctx_offset
            )
            row += _draw_context(stdscr, ctx_slice, current_in_slice, row, ctx_budget, width, show_scroll=True)
        else:
            # Fallback: neighbours within the gt sample itself (offset applied)
            center = state.index + state.ctx_offset
            center = max(CONTEXT_WINDOW, min(len(messages) - 1 - CONTEXT_WINDOW, center))
            start = max(0, center - CONTEXT_WINDOW)
            end = min(len(messages), center + CONTEXT_WINDOW + 1)
            ctx_slice = messages[start:end]
            current_in_slice = state.index - start
            if not (0 <= current_in_slice < len(ctx_slice)):
                current_in_slice = -1
            row += _draw_context(stdscr, ctx_slice, current_in_slice, row, ctx_budget, width, show_scroll=True)

        row += 1  # blank separator
        row += _draw_message(stdscr, current_msg, row, width)
        row += _draw_step_indicator(stdscr, state.step, row, width)

        _draw_controls(stdscr, state.step, last_ann, row + 1, width)

        stdscr.refresh()

        key = stdscr.getch()
        new_state = _handle_key(key, state, total, has_context)

        if new_state.should_save and not new_state.should_quit and save_callback is not None:
            save_callback(list(new_state.annotations))
            new_state = replace(new_state, should_save=False)

        # Patch row_id into the freshly added annotation
        if len(new_state.annotations) > len(state.annotations):
            latest = new_state.annotations[-1]
            if latest["row_id"] is None:
                current_rid = int(current_msg["row_id"])
                patched = {**latest, "row_id": current_rid}
                # Remove any prior annotation for this row_id before appending the new one
                # so navigating back and re-annotating replaces rather than duplicates.
                deduped = tuple(a for a in new_state.annotations[:-1] if a.get("row_id") != current_rid)
                new_state = replace(
                    new_state,
                    annotations=deduped + (patched,),
                )

        state = new_state
