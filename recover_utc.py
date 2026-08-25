import random
import pandas as pd

FPS = 30

# Range to draw synthetic "new task" start times from.
RANGE_START = pd.Timestamp("2026-01-01", tz="UTC")
RANGE_END = pd.Timestamp("2026-05-01", tz="UTC")

current_timestamp = None


def get_random_utc_timestamp_in_range():
    """Pick a uniformly random UTC timestamp between RANGE_START and RANGE_END."""
    span_seconds = (RANGE_END - RANGE_START).total_seconds()
    offset_seconds = random.uniform(0, span_seconds)
    return RANGE_START + pd.Timedelta(seconds=offset_seconds)

def get_utc_starttime():
    return pd.Timestamp("2026-01-01", tz="UTC")

def set_to_zero_utc_timestamp() -> float:
    return pd.Timestamp("2026-01-01", tz="UTC").timestamp()

def recover_utc_timestamp(task_annotation: str, frame_cnt: int) -> float:
    """
    Synthesize a plausible UTC start timestamp for an episode.

    - First call ever: pick a random timestamp in RANGE_START..RANGE_END and use it.
    - On each subsequent call, ask the operator whether this episode is a new task
      ("y" => reset to a new random timestamp, since we have no way to know the
      real gap between unrelated recording sessions).
    - Otherwise, assume this episode was recorded back-to-back with the previous
      one and advance the clock by frame_cnt / FPS seconds.

    Returns UTC seconds since epoch (float), matching the schema.
    """
    global current_timestamp

    if current_timestamp is None:
        current_timestamp = get_utc_starttime()
        print(f"\n======{task_annotation} {current_timestamp}======")
        return current_timestamp.timestamp()

    if input(f'"{task_annotation}" - new task? (y/n): ').strip().lower() == "y":
        print(f"\n======{task_annotation} {current_timestamp}======")
        current_timestamp = get_utc_starttime()
        return current_timestamp.timestamp()

    
    current_timestamp = current_timestamp + pd.Timedelta(seconds=frame_cnt / FPS)
    return current_timestamp.timestamp()
import json
import subprocess
from datetime import datetime, timezone


from pathlib import Path


def get_exact_original_paths(
    root_path: str | Path
) -> list[str]:
    root = Path(root_path)
    delphi_episode_paths = []

    for path in root.iterdir():
        if not path.is_file():
            continue
        path = path.with_suffix("")
        if path in delphi_episode_paths:
            continue
        delphi_episode_paths.append(path)

    return [str(p) for p in delphi_episode_paths]

def get_video_utc_mtime(file_path: str | Path) -> int:
    """Returns the UTC Unix timestamp (integer) for a file on Linux using st_mtime.

    Appends .mp4 if no extension is present.
    """
    path = Path(file_path)

    # Ensure .mp4 extension if missing
    if not path.suffix:
        path = path.with_suffix(".mp4")

    # Get POSIX modification time (seconds since epoch in UTC)
    mtime_seconds = path.stat().st_mtime

    # Convert to explicit UTC datetime integer
    dt_utc = datetime.fromtimestamp(mtime_seconds, tz=timezone.utc)
    return int(dt_utc.timestamp())



def get_all_utc_start_times(paths: list[str]) -> list[int]:
    """Processes a list of path strings and returns their UTC start timestamps."""
    return [get_video_utc_mtime(p) for p in paths]