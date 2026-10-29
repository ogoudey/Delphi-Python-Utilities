"""
Delphi compound-task folders  =>  LeRobotDataset v3.0 (+ language_persistent subtasks).

Input layout (V3_Delphi27_CompoundTaskFolders):

    root/
      <Compound task>/            e.g. "Making coffee"
        <uuid>.mp4                egocentric head video (VFR, ~17-23 fps)
        <uuid>.csv                head + right-hand joint poses (~23 Hz, world frame)
        <uuid>.json               subtask spans [{startTime, endTime, taskId, transcription}, ...]

Output (one LeRobot episode per recording):
    task                  -> overall task, e.g. "I make coffee"           (meta/tasks.parquet)
    language_persistent   -> one style="subtask" row per JSON span, stamped at its start
    observation.images.head, observation.state, observation.hand_tracked, action (= next state)
    meta/delphi_episodes.jsonl -> per-episode provenance (source uuid, clock offset, raw spans)

Clocks. The three files do not share a t=0:
    * CSV timestamps are absolute UTC.  CSV clock := seconds since the first CSV row.
    * The MP4 carries no absolute time, and in many recordings starts a few seconds after
      the CSV (the offset is NOT derivable from durations).  We recover it per episode by
      cross-correlating head angular speed (CSV quaternions) against global image motion
      (phase correlation between consecutive frames).  csv_time = video_time + lag.
    * JSON subtask times are on the CSV clock (see `json_clock_check` in the dry run).
The output timeline is the video clock resampled to a constant `fps`, clipped to the
span where both video and CSV exist.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import av
import numpy as np
import pandas as pd

from dataset_processor import V3_Delphi27_CompoundTaskFolders

COMPOUND_TASK_TO_TASK = {
    "Making coffee": "I make coffee",
    "Bagging groceries": "I bag groceries",
    "Putting items away": "I put items away",
}

HEAD_JOINT = "head"
CAMERA_NAME = "head"
# Order of the 26 hand joints as they appear in the Delphi CSV (OpenXR hand-joint order).
HAND_JOINTS = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip",
]

# Alignment acceptance: peak correlation and margin over the best peak >0.5s away.
MIN_ALIGN_R = 0.35
MIN_ALIGN_MARGIN = 0.10
ALIGNMENT_OVERRIDES = Path(__file__).with_name("alignment_overrides.json")   # uuid -> note


@dataclass
class ConverterConfig:
    fps: int = 30
    resize: Optional[tuple] = None           # (width, height) or None for native
    lag_search_s: tuple = (-3.0, 12.0)
    hands: tuple = ("right",)                # handedness present in this data
    robot_type: str = "human"


@dataclass
class Alignment:
    lag: float                 # csv_time = video_time + lag
    r: float                   # peak normalized correlation
    second_r: float            # best correlation > 0.5 s away from the peak
    lag_first_half: float
    lag_second_half: float
    video_first_pts: float
    video_last_pts: float
    n_video_frames: int

    @property
    def ok(self) -> bool:
        return self.r >= MIN_ALIGN_R and (self.r - self.second_r) >= MIN_ALIGN_MARGIN


@dataclass
class RecordingPlan:
    """Everything needed to write one episode, minus the pixels."""
    stem: Path
    compound_task: str
    task: str
    alignment: Alignment
    t_start: float             # video clock, first output frame
    n_frames: int
    subtasks: List[dict] = field(default_factory=list)   # episode-clock spans
    dropped_subtasks: List[dict] = field(default_factory=list)
    hand_tracked_frac: float = 0.0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_recordings(root: Path) -> List[Path]:
    """<root>/<compound task>/<uuid>.{mp4,csv,json} -> sorted list of <root>/<task>/<uuid>."""
    stems = []
    for subdir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for stem in sorted({p.with_suffix("") for p in subdir.iterdir()
                            if p.is_file() and not p.name.startswith(".")}):
            missing = [s for s in (".mp4", ".csv", ".json") if not stem.with_suffix(s).exists()]
            if missing:
                print(f"[skip] {stem}: missing {missing}")
                continue
            stems.append(stem)
    return stems


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def load_tracking(csv_path: Path, cfg_in=V3_Delphi27_CompoundTaskFolders) -> dict:
    """
    Returns {joint_name: (times[N], pose[N,7])} on the CSV clock, with joint names
    merged with handedness ("RightWrist") as in the existing pipeline. Quaternions are
    made sign-continuous over time (q and -q are the same rotation; flips break learning).
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    ts = pd.to_datetime(df[cfg_in.joints_timestamp_col], utc=True)
    t_abs = (ts - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
    df["t"] = t_abs - t_abs.min()

    source = df[cfg_in.joints_source_col].str.strip().str.lower()
    joint = df[cfg_in.joints_joint_col].str.strip()
    df["name"] = np.where(source.isin(["right", "left"]), source.str.capitalize() + joint, joint)

    cols = cfg_in.joints_position_cols + cfg_in.joints_rotation_cols
    out = {}
    for name, g in df.groupby("name", sort=False):
        g = g.drop_duplicates("t").sort_values("t")
        pose = np.array(g[cols].to_numpy(dtype=np.float64), copy=True)
        q = pose[:, 3:]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        out[name] = (np.array(g["t"].to_numpy(), copy=True), pose)
    out["_all_times"] = np.unique(df["t"].to_numpy())
    return out


def head_angular_speed(tracking: dict):
    t, pose = tracking[HEAD_JOINT]
    q = pose[:, 3:]
    d = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0, 1)
    return t[1:], 2 * np.arccos(d) / np.maximum(np.diff(t), 1e-3)


def sample_pose(t_query: np.ndarray, t: np.ndarray, pose: np.ndarray, max_gap: float):
    """
    Linear interpolation of position + normalized-lerp of (sign-continuous) quaternion.
    `valid[i]` is False when t_query[i] falls in a tracking gap longer than `max_gap`
    or outside the joint's observed time range; values there are still filled (held /
    interpolated) so the vector is never NaN, and the validity is exported separately.
    """
    out = np.empty((len(t_query), 7))
    for k in range(7):
        out[:, k] = np.interp(t_query, t, pose[:, k])
    out[:, 3:] /= np.linalg.norm(out[:, 3:], axis=1, keepdims=True)

    idx = np.searchsorted(t, t_query)
    left = t[np.clip(idx - 1, 0, len(t) - 1)]
    right = t[np.clip(idx, 0, len(t) - 1)]
    inside = (t_query >= t[0]) & (t_query <= t[-1])
    gap = np.where(inside, right - left, np.inf)
    return out, inside & (gap <= max_gap)


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def video_global_motion(mp4: Path, w: int = 160, h: int = 90):
    """Per-frame global image speed (px/s at w x h) via phase correlation; plus all pts."""
    import cv2

    win = cv2.createHanningWindow((w, h), cv2.CV_64F)
    times, speed, pts_all = [], [], []
    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        prev = None
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            pts_all.append(t)
            g = f.reformat(width=w, height=h, format="gray").to_ndarray().astype(np.float64)
            if prev is not None:
                (dx, dy), _ = cv2.phaseCorrelate(prev, g, win)
                times.append(t)
                speed.append(np.hypot(dx, dy) / max(t - pt, 1e-3))
            prev, pt = g, t
    return np.array(times), np.array(speed), np.array(pts_all)


def _xcorr(tv, sv, tc, sc, lags, fs=30.0, t_range=None):
    lo, hi = t_range if t_range else (tv[0], tv[-1])
    grid = np.arange(lo, hi, 1 / fs)
    v = np.interp(grid, tv, sv)
    r = np.full(len(lags), np.nan)
    for i, L in enumerate(lags):
        g = grid + L
        m = (g >= tc[0]) & (g <= tc[-1])
        if m.sum() < 0.6 * len(grid):
            continue
        a = v[m]
        b = np.interp(g[m], tc, sc)
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        r[i] = float(np.mean(a * b))
    return r


def estimate_alignment(stem: Path, cfg: ConverterConfig) -> Alignment:
    tv, sv, pts = video_global_motion(stem.with_suffix(".mp4"))
    tc, sc = head_angular_speed(load_tracking(stem.with_suffix(".csv")))
    lags = np.arange(cfg.lag_search_s[0], cfg.lag_search_s[1], 1 / 60)

    def best(t_range=None):
        r = _xcorr(tv, sv, tc, sc, lags, t_range=t_range)
        if np.all(np.isnan(r)):
            return np.nan, np.nan, np.nan
        i = int(np.nanargmax(r))
        r2 = r.copy()
        r2[np.abs(lags - lags[i]) < 0.5] = np.nan
        # sub-sample refinement (parabolic) on the coarse peak
        lag = lags[i]
        if 0 < i < len(r) - 1 and not np.isnan(r[i - 1]) and not np.isnan(r[i + 1]):
            denom = r[i - 1] - 2 * r[i] + r[i + 1]
            if denom < 0:
                lag += 0.5 * (r[i - 1] - r[i + 1]) / denom * (lags[1] - lags[0])
        return float(lag), float(r[i]), float(np.nanmax(r2))

    lag, r, r2 = best()
    mid = 0.5 * (tv[0] + tv[-1])
    lag_a, _, _ = best((tv[0], mid))
    lag_b, _, _ = best((mid, tv[-1]))
    return Alignment(lag=lag, r=r, second_r=r2, lag_first_half=lag_a, lag_second_half=lag_b,
                     video_first_pts=float(pts[0]), video_last_pts=float(pts[-1]),
                     n_video_frames=int(len(pts)))


def load_or_compute_alignments(stems: List[Path], cache_path: Path, cfg: ConverterConfig,
                               workers: int) -> dict:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [s for s in stems if s.name not in cache]
    if todo:
        print(f"[align] estimating clock offsets for {len(todo)} recording(s) with {workers} worker(s)...")
        from concurrent.futures import as_completed
        failed = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(estimate_alignment, s, cfg): s for s in todo}
            for n, fut in enumerate(as_completed(futs), 1):
                s = futs[fut]
                try:
                    a = fut.result()
                except Exception as e:
                    failed.append(s)
                    print(f"[align] {n}/{len(todo)} {s.parent.name}/{s.name[:8]} FAILED: {e!r}", flush=True)
                    continue
                cache[s.name] = asdict(a)
                print(f"[align] {n}/{len(todo)} {s.parent.name}/{s.name[:8]} lag={a.lag:+.3f}s "
                      f"r={a.r:.2f} (2nd {a.second_r:.2f}) halves={a.lag_first_half:+.2f}/{a.lag_second_half:+.2f}"
                      f"{'' if a.ok else '  <-- LOW CONFIDENCE'}", flush=True)
                cache_path.write_text(json.dumps(cache, indent=1))
        if failed:
            raise RuntimeError(f"alignment failed for {len(failed)} recording(s): {[s.name for s in failed]}")
    return {s.name: Alignment(**cache[s.name]) for s in stems}


# ---------------------------------------------------------------------------
# Planning (shared by dry run and real run)
# ---------------------------------------------------------------------------

def plan_recording(stem: Path, alignment: Alignment, tracking: dict, cfg: ConverterConfig) -> RecordingPlan:
    compound = stem.parent.name
    if compound not in COMPOUND_TASK_TO_TASK:
        raise KeyError(f"No task string for compound task folder {compound!r}; add it to COMPOUND_TASK_TO_TASK")

    csv_t = tracking["_all_times"]
    lag = alignment.lag
    # Overlap of [video] and [csv mapped to video clock]
    t_start = max(alignment.video_first_pts, csv_t[0] - lag)
    t_end = min(alignment.video_last_pts, csv_t[-1] - lag)
    n_frames = int(np.floor((t_end - t_start) * cfg.fps)) + 1

    with stem.with_suffix(".json").open() as f:
        raw = json.load(f)

    ep_end = (n_frames - 1) / cfg.fps
    subtasks, dropped = [], []
    for s in sorted(raw, key=lambda s: s["startTime"]):
        # CSV clock -> video clock -> episode clock
        start = s["startTime"] - lag - t_start
        end = s["endTime"] - lag - t_start
        span = {"start_s": start, "end_s": end, "text": s.get("transcription", ""),
                "task_id": s.get("taskId"), "source_span_id": s.get("id"),
                "source_tracked": s.get("tracked"),
                "source_start_s": s["startTime"], "source_end_s": s["endTime"]}
        if not span["text"] or end <= 0 or start >= ep_end:
            dropped.append(span)
            continue
        span["start_s"] = max(0.0, start)
        span["end_s"] = min(ep_end, end)
        subtasks.append(span)

    t_frames = t_start + np.arange(n_frames) / cfg.fps + lag
    hand_valid = np.ones(n_frames, bool)
    for hand in cfg.hands:
        for j in HAND_JOINTS:
            name = hand.capitalize() + j
            _, v = sample_pose(t_frames, *tracking[name], max_gap=0.1)
            hand_valid &= v

    return RecordingPlan(stem=stem, compound_task=compound, task=COMPOUND_TASK_TO_TASK[compound],
                         alignment=alignment, t_start=t_start, n_frames=n_frames,
                         subtasks=subtasks, dropped_subtasks=dropped,
                         hand_tracked_frac=float(hand_valid.mean()))


POSE_FIELDS = ["position.x", "position.y", "position.z",
               "orientation.qx", "orientation.qy", "orientation.qz", "orientation.qw"]


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()   # "IndexTip" -> "index_tip"


def state_names(cfg: ConverterConfig) -> List[str]:
    """e.g. "head.position.x", "right_hand.index_tip.orientation.qw" (7 values per joint)."""
    joints = [HEAD_JOINT] + [f"{h}_hand.{_snake(j)}" for h in cfg.hands for j in HAND_JOINTS]
    return [f"{j}.{f}" for j in joints for f in POSE_FIELDS]


def build_state(plan: RecordingPlan, tracking: dict, cfg: ConverterConfig):
    """(n_frames, D) float32 state and (n_frames, n_hands) float32 tracked flags."""
    t = plan.t_start + np.arange(plan.n_frames) / cfg.fps + plan.alignment.lag
    head, _ = sample_pose(t, *tracking[HEAD_JOINT], max_gap=np.inf)
    blocks, flags = [head], []
    for hand in cfg.hands:
        valid = np.ones(len(t), bool)
        for j in HAND_JOINTS:
            p, v = sample_pose(t, *tracking[hand.capitalize() + j], max_gap=0.1)
            blocks.append(p)
            valid &= v
        flags.append(valid)
    state = np.concatenate(blocks, axis=1).astype(np.float32)
    return state, np.stack(flags, axis=1).astype(np.float32)


def iter_resampled_frames(plan: RecordingPlan, cfg: ConverterConfig):
    """
    VFR -> CFR: output frame k shows the source frame on screen at t_start + k/fps
    (latest frame with pts <= t). Yields HxWx3 uint8 arrays.
    """
    targets = plan.t_start + np.arange(plan.n_frames) / cfg.fps
    eps = 1e-4
    k = 0
    held = None
    with av.open(str(plan.stem.with_suffix(".mp4"))) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        kw = {"format": "rgb24"}
        if cfg.resize:
            kw.update(width=cfg.resize[0], height=cfg.resize[1])
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            # Emit every target whose display time is before this frame's pts.
            while k < plan.n_frames and targets[k] + eps < t:
                if held is None:   # target precedes first frame (only possible by rounding)
                    held = f.to_ndarray(**kw)
                yield held
                k += 1
            if k >= plan.n_frames:
                return
            held = f.to_ndarray(**kw)
        while k < plan.n_frames:   # targets at/after the last pts
            yield held
            k += 1


def video_shape(stem: Path, cfg: ConverterConfig):
    if cfg.resize:
        return (cfg.resize[1], cfg.resize[0], 3)
    with av.open(str(stem.with_suffix(".mp4"))) as c:
        s = c.streams.video[0]
        return (s.height, s.width, 3)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def make_features(shape, cfg: ConverterConfig) -> dict:
    names = state_names(cfg)
    return {
        f"observation.images.{CAMERA_NAME}": {"dtype": "video", "shape": shape,
                                              "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.hand_tracked": {"dtype": "float32", "shape": (len(cfg.hands),),
                                     "names": [f"{h}_hand" for h in cfg.hands]},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }


def write_recording(dataset, plan: RecordingPlan, cfg: ConverterConfig) -> int:
    tracking = load_tracking(plan.stem.with_suffix(".csv"))
    state, tracked = build_state(plan, tracking, cfg)
    action = np.concatenate([state[1:], state[-1:]], axis=0)   # action[t] = state[t+1]
    n = 0
    for k, img in enumerate(iter_resampled_frames(plan, cfg)):
        dataset.add_frame({
            f"observation.images.{CAMERA_NAME}": img,
            "observation.state": state[k],
            "observation.hand_tracked": tracked[k],
            "action": action[k],
            "task": plan.task,
        })
        n += 1
    if n != plan.n_frames:
        raise RuntimeError(f"{plan.stem}: expected {plan.n_frames} frames, decoded {n}")
    dataset.save_episode()
    return n


def write_language_and_provenance(root: Path, plans: List[RecordingPlan], fps: int):
    """Stamp subtask rows into language_persistent via lerobot's own writer, and keep
    meta/info.json in sync exactly as `lerobot-annotate` does."""
    import tempfile
    from lerobot.annotations.steerable_pipeline.reader import iter_episodes, snap_to_frame
    from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
    from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter
    from lerobot.datasets.io_utils import load_info, write_info
    from lerobot.datasets.language import language_feature_info

    records = list(iter_episodes(root))
    if len(records) != len(plans):
        raise RuntimeError(f"dataset has {len(records)} episodes but {len(plans)} plans")

    with tempfile.TemporaryDirectory(dir=root) as staging_dir:
        staging_dir = Path(staging_dir)
        for record, plan in zip(records, plans):
            rows, seen = [], set()
            for span in plan.subtasks:
                ts = snap_to_frame(span["start_s"], record.frame_timestamps)
                key = (ts, span["text"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"role": "assistant", "content": span["text"], "style": "subtask",
                             "timestamp": ts, "camera": None, "tool_calls": None})
            EpisodeStaging(staging_dir, record.episode_index).write("plan", rows)
        LanguageColumnsWriter().write_all(records, staging_dir, root)

    info = load_info(root)
    features = {**info.features, **language_feature_info()}
    for key, text in FEATURE_DESCRIPTIONS.items():
        if key in features:
            features[key] = {**features[key], "description": text}
    info.features = features
    write_info(info, root)

    overrides = json.loads(ALIGNMENT_OVERRIDES.read_text()) if ALIGNMENT_OVERRIDES.exists() else {}
    with (root / "meta" / "delphi_episodes.jsonl").open("w") as f:
        for record, plan in zip(records, plans):
            a = plan.alignment
            f.write(json.dumps({
                "episode_index": record.episode_index,
                "task": plan.task,
                "source_folder": plan.compound_task,
                "source_recording_id": plan.stem.name,
                "duration_s": plan.n_frames / fps,
                "episode_start_in_source_video_s": plan.t_start,
                "csv_time_minus_video_time_s": a.lag,
                "alignment": {
                    "method": "xcorr of head angular speed (csv) vs global image motion (video)",
                    "peak_correlation": a.r,
                    "second_peak_correlation": a.second_r,
                    "lag_first_half_s": a.lag_first_half,
                    "lag_second_half_s": a.lag_second_half,
                    "manual_override": overrides.get(plan.stem.name),
                    "source_video_first_pts_s": a.video_first_pts,
                    "source_video_last_pts_s": a.video_last_pts,
                    "source_video_frames": a.n_video_frames,
                },
                "right_hand_tracked_fraction": plan.hand_tracked_frac,
                "subtasks": plan.subtasks,
                "dropped_subtasks": plan.dropped_subtasks,
            }) + "\n")

    (root / "README.md").write_text(dataset_card(plans, fps))


FEATURE_DESCRIPTIONS = {
    "observation.images.head": "Egocentric RGB video from the head-mounted camera, resampled to constant fps "
                               "(each frame = the source frame on screen at that instant).",
    "observation.state": "Tracked poses at this frame: head, then 26 right-hand joints (OpenXR joint order). "
                         "7 values per joint: position x,y,z (metres, tracking-space frame) and orientation "
                         "quaternion qx,qy,qz,qw (unit norm, sign-continuous over time). Hand values are "
                         "interpolated/held while untracked; check observation.hand_tracked.",
    "observation.hand_tracked": "1.0 if the right hand was tracked at this frame (no tracking gap > 0.1 s), "
                                "else 0.0. When 0.0 the hand part of state/action is not a real measurement.",
    "action": "Same layout as observation.state, taken from the next frame: action[t] = state[t+1]. "
              "The last frame of each episode repeats its own state.",
    "language_persistent": "Subtask narration from the source annotations: one row per span with "
                           "style='subtask', role='assistant', timestamp = span start (episode seconds). "
                           "A subtask stays active until the next one starts. Same list on every frame.",
    "language_events": "Unused (always empty).",
    "task_index": "Index into meta/tasks.parquet: the overall task of the episode "
                  "('I bag groceries', 'I make coffee', 'I put items away').",
    "timestamp": "Seconds since the start of the episode (frame_index / fps).",
}


def dataset_card(plans: List[RecordingPlan], fps: int) -> str:
    n_frames = sum(p.n_frames for p in plans)
    per_task = {t: sum(p.task == t for p in plans) for t in sorted({p.task for p in plans})}
    feature_rows = "\n".join(f"| `{k}` | {v} |" for k, v in FEATURE_DESCRIPTIONS.items())
    task_rows = "\n".join(f"| {t} | {n} |" for t, n in per_task.items())
    return f"""---
tags: [LeRobot, egocentric, human-demonstration, hand-tracking]
---
# Delphi egocentric household tasks (LeRobot v3.0)

Human demonstrations recorded with a head-mounted camera plus head and right-hand
tracking. One episode per source recording. {len(plans)} episodes, {n_frames} frames,
{n_frames / fps / 3600:.2f} h at {fps} fps. `robot_type` is `human`: there is no robot, so
`action` is the next tracked pose, not a motor command.

## Tasks (`meta/tasks.parquet`)

Every frame's `task` is the overall task of its episode:

| task | episodes |
|---|---|
{task_rows}

## Subtasks (`language_persistent`)

Each episode carries the annotated subtasks as `style="subtask"` rows, e.g.
"I place a coffee capsule into the coffee machine". To get the active subtask at time t,
take the last row with `timestamp <= t`. Original span start/end times, annotation
task ids and span ids are in `meta/delphi_episodes.jsonl`.

## Features

| feature | meaning |
|---|---|
{feature_rows}

`observation.state` / `action` element names look like `head.position.x` and
`right_hand.index_tip.orientation.qw`; the full list is in `meta/info.json`.
The left hand was not recorded. The tracking origin and handedness of the coordinate
frame are those of the source tracker; positions are in metres.

## Time alignment

The source video, tracking CSV and annotation JSON do not share a start time. For each
recording the offset between video and CSV was estimated by cross-correlating head
rotation speed with image motion (median residual well under one frame). Annotation
times are on the CSV clock and were shifted onto the episode clock. The video's variable
frame rate (17-23 fps) was resampled to a constant {fps} fps. Per-episode offsets,
alignment confidence and any manual override are recorded in `meta/delphi_episodes.jsonl`.

## `meta/delphi_episodes.jsonl`

One JSON object per episode: `episode_index`, `task`, `source_folder`,
`source_recording_id` (source file stem), `duration_s`,
`episode_start_in_source_video_s`, `csv_time_minus_video_time_s`, `alignment` (method and
correlation diagnostics), `right_hand_tracked_fraction`, and `subtasks` / `dropped_subtasks`.
Each subtask has `start_s` / `end_s` (episode seconds, clipped to the episode), `text`,
`task_id`, `source_span_id`, `source_tracked`, and `source_start_s` / `source_end_s`
(unshifted times from the source JSON).
"""


def json_clock_check(stems: List[Path], alignments: dict) -> dict:
    """
    Are JSON span times on the CSV clock or the video clock? Annotators watched the
    video, so no span can end after the last video frame. Count episodes whose last
    span overruns the video end under each hypothesis (only where |lag| > 1 s, i.e.
    where the hypotheses differ). On this data the CSV clock gives 0 overruns.
    """
    over_csv = over_vid = n = 0
    for stem in stems:
        a = alignments[stem.name]
        spans = json.loads(stem.with_suffix(".json").read_text())
        if abs(a.lag) <= 1.0 or not spans:
            continue
        n += 1
        end = max(s["endTime"] for s in spans)
        over_csv += (end - a.lag) > a.video_last_pts + 0.2
        over_vid += end > a.video_last_pts + 0.2
    return {"n": n, "overruns_if_csv_clock": int(over_csv), "overruns_if_video_clock": int(over_vid)}


def _parse_resize(s: str | None):
    if not s:
        return None
    w, h = s.lower().split("x")
    return (int(w), int(h))


def check_encoder_works(encoder, shape):
    """Encode a few frames with exactly the options LeRobot will use, so a missing GPU or
    a rejected option fails here, not an hour into the run. (PyAV lists h264_nvenc even
    when no NVIDIA device is reachable.)"""
    import io
    from fractions import Fraction
    opts = encoder.get_codec_options(as_strings=True)
    out = av.open(io.BytesIO(), "w", format="mp4")
    try:
        st = out.add_stream(encoder.vcodec, rate=30, options=opts)
        st.pix_fmt, st.height, st.width = encoder.pix_fmt, shape[0], shape[1]
        st.time_base = Fraction(1, 30)
        for k in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros(shape, np.uint8), format="rgb24")
            frame.pts = k
            list(st.encode(frame))
        list(st.encode())
    except Exception as e:
        raise SystemExit(f"Video encoder {encoder.vcodec!r} with options {opts} is not usable "
                         f"on this machine: {e}") from e
    finally:
        out.close()


def run(source: Path, destination: Path, repo_id: str, *, dry_run: bool, limit_per_task: int | None,
        fps: int, resize: str | None, workers: int, alignment_cache: Path, allow_low_confidence: bool,
        vcodec: str, crf: int | None, encoder_options: dict | None = None):
    cfg = ConverterConfig(fps=fps, resize=_parse_resize(resize))
    stems = find_recordings(source)
    if limit_per_task:
        by_task = {}
        for s in stems:
            by_task.setdefault(s.parent.name, []).append(s)
        stems = [s for group in by_task.values() for s in group[:limit_per_task]]
    print(f"{source} => {destination}  ({len(stems)} recordings, fps={cfg.fps}, resize={cfg.resize})")

    alignments = load_or_compute_alignments(stems, alignment_cache, cfg, workers)

    plans, bad = [], []
    for stem in stems:
        tracking = load_tracking(stem.with_suffix(".csv"))
        plan = plan_recording(stem, alignments[stem.name], tracking, cfg)
        a = plan.alignment
        drift = a.lag_second_half - a.lag_first_half
        flag = "" if a.ok else "  LOW-CONFIDENCE ALIGNMENT"
        if abs(drift) > 0.2:
            flag += f"  DRIFT {drift:+.2f}s"
        print(f"  {plan.compound_task[:18]:18s} {stem.name[:8]} lag={a.lag:+.3f}s r={a.r:.2f} "
              f"frames={plan.n_frames} ({plan.n_frames / cfg.fps:.1f}s) subtasks={len(plan.subtasks)} "
              f"dropped={len(plan.dropped_subtasks)} hand={plan.hand_tracked_frac:.0%}{flag}")
        if not a.ok:
            bad.append(stem)
        plans.append(plan)

    total = sum(p.n_frames for p in plans)
    print(f"\n{len(plans)} episodes, {total} frames, {total / cfg.fps / 3600:.2f} h at {cfg.fps} fps")
    print(f"tasks: { {t: sum(p.task == t for p in plans) for t in sorted({p.task for p in plans})} }")
    if bad:
        print(f"{len(bad)} low-confidence alignment(s): {[s.name[:8] for s in bad]}")

    if dry_run:
        print("\n[json clock check]", json_clock_check(stems, alignments))
        print("\nDry run: nothing written.")
        return

    if bad and not allow_low_confidence:
        raise SystemExit("Refusing to write with low-confidence alignments (pass --allow-low-confidence).")

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    enc_kwargs = {"vcodec": vcodec, "extra_options": dict(encoder_options or {})}
    if vcodec.endswith("_nvenc"):
        # LeRobot's GOP of 2 leaves no room for NVENC's default B-frames: avcodec_open2 fails
        # with EINVAL (seen on a GTX 1060). All-I/P frames are what g=2 means anyway.
        enc_kwargs["extra_options"].setdefault("bf", "0")
    if crf is not None:
        enc_kwargs["crf"] = crf
    encoder = RGBEncoderConfig(**enc_kwargs)
    shape = video_shape(stems[0], cfg)
    check_encoder_works(encoder, shape)
    dataset = LeRobotDataset.create(
        repo_id=repo_id, root=destination, fps=cfg.fps, robot_type=cfg.robot_type,
        features=make_features(shape, cfg), use_videos=True,
        streaming_encoding=True, rgb_encoder=encoder,
    )
    import time
    t0 = time.time()
    done = 0
    for i, plan in enumerate(plans):
        n = write_recording(dataset, plan, cfg)
        done += n
        el = time.time() - t0
        print(f"[write] {i + 1}/{len(plans)} {plan.compound_task}/{plan.stem.name[:8]} {n} frames | "
              f"{done / el:.1f} fps overall, ETA {(total - done) / max(done / el, 1e-6) / 60:.1f} min", flush=True)
    dataset.finalize()

    write_language_and_provenance(destination, plans, cfg.fps)
    print(f"Done: {destination}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-per-task", type=int, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--resize", default=None, help="WxH, e.g. 640x360 (default: native)")
    ap.add_argument("--workers", type=int, default=8, help="processes for alignment estimation")
    ap.add_argument("--alignment-cache", type=Path, default=Path("alignment_cache.json"))
    ap.add_argument("--allow-low-confidence", action="store_true")
    ap.add_argument("--vcodec", default="h264", help="h264 (CPU) | h264_nvenc (NVIDIA GPU) | libsvtav1 ...")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--encoder-option", action="append", default=[], metavar="KEY=VALUE",
                    help="extra codec option passed to the encoder, e.g. bf=0 (repeatable)")
    a = ap.parse_args()
    run(a.source, a.destination, a.repo_id or f"local/{a.destination.name}", dry_run=a.dry_run,
        limit_per_task=a.limit_per_task, fps=a.fps, resize=a.resize, workers=a.workers,
        alignment_cache=a.alignment_cache, allow_low_confidence=a.allow_low_confidence,
        vcodec=a.vcodec, crf=a.crf,
        encoder_options=dict(o.split("=", 1) for o in a.encoder_option))
