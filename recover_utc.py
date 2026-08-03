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