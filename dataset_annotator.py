#!/usr/bin/env python
"""Write custom language_persistent annotations into a LeRobot v3.0/v3.1 dataset.

This reuses `LanguageColumnsWriter` from the built-in steerable annotation
pipeline (lerobot.annotations.steerable_pipeline) so the parquet rewrite,
struct schema, and meta/info.json sync all match what `lerobot-annotate`
itself produces -- we just skip the VLM modules and stage our own rows.

language_events is left untouched (== `[]` for every frame), since we never
stage anything under the "interjections" / "vqa" module buckets.

Expected raw annotation layout:

    raw_data_path/
      <compound_task>/
        <uuid>_<episode_name>.json   # list of subtask dicts, e.g.:
          [
            {
              "id": "...", "startTime": 3.69, "endTime": 9.868,
              "taskId": "PutGroceryItemInBag",
              "transcription": "I place spice from the countertop into the bag",
              "tracked": true
            },
            ...
          ]

EPISODE MATCHING -- two strategies:
   Match by (task == compound_task subdir name) + closest episode
   duration, within a task group. Requires the per-task annotation-file
   count to equal the per-task episode count, and that the closest-duration
   candidate beats the second-closest by a configurable margin -- anything
   ambiguous is reported and refused rather than guessed. Duration is
   approximated as `max(subtask endTime)` on the annotation side, which can
   under-count true episode length by however much untranscribed trailing
   idle time exists -- pad --duration-tolerance accordingly for your data.

Usage:
    python annotate_custom_language.py DATASET_ROOT RAW_DATA_PATH [--creation-log LOG] [--dry-run]

DATASET_ROOT is modified in place. Back up the dataset (or work on a copy)
before running -- the writer rewrites data/chunk-*/file-*.parquet in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import av

from lerobot.annotations.steerable_pipeline.reader import (
    EpisodeRecord,
    iter_episodes,
    snap_to_frame,
)
from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
from lerobot.annotations.steerable_pipeline.validator import StagingValidator
from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter
from lerobot.datasets.io_utils import load_info, write_info
from lerobot.datasets.language import SAY_TOOL_SCHEMA, language_feature_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class DatasetAnnotator:
    """Stages custom subtask annotations and runs them through the real writer."""

    def __init__(self, style: str = "subtask", role: str = "assistant") -> None:
        # "subtask" is a PERSISTENT_STYLES entry -> routes to language_persistent.
        # Swap to "plan" or "memory" if that fits your annotations better; any
        # of PERSISTENT_STYLES works the same way through the writer.
        self.style = style
        self.role = role

    # ------------------------------------------------------------------
    # 1. Read your raw JSON annotations
    # ------------------------------------------------------------------
    def _load_raw_annotations(self, raw_data_path: Path) -> dict[str, dict[str, Any]]:
        """Returns {sort_key: {"compound_task": ..., "subtasks": [...]}}."""
        per_episode: dict[str, dict[str, Any]] = {}
        for subdir in sorted(raw_data_path.iterdir()):
            if not subdir.is_dir():
                continue
            compound_task = subdir.name
            for path in sorted(subdir.iterdir()):
                if not path.is_file() or path.suffix != ".json":
                    continue
                with path.open(encoding="utf-8") as f:
                    subtasks = json.load(f)
                sort_key = f"{path.stem}"
                per_episode[sort_key] = {
                    "compound_task": compound_task,
                    "source_file": path,
                    "subtasks": subtasks,
                }
        return per_episode


    # ------------------------------------------------------------------
    # 3. Match via (task, closest duration)
    # ------------------------------------------------------------------
    @staticmethod
    def _episode_duration(record: EpisodeRecord) -> float:
        ts = record.frame_timestamps
        return float(ts[-1]) - float(ts[0]) if ts else 0.0

    @staticmethod
    def _raw_duration(entry: dict[str, Any]) -> float:
        subtasks = entry["subtasks"]
        return max((float(s["endTime"]) for s in subtasks), default=0.0)

    def _match_via_duration(
            self,
            dataset: LeRobotDataset,  # Loaded local LeRobotDataset object
            durations_per_episodes: dict[str, float], # {episode_stem: duration_in_seconds}
            raw: dict[str, dict[str, Any]],           # {raw_key: raw_annotation_dict}
            duration_tolerance: float = 0.5,
        ) -> list[tuple[dict[str, Any], int]]:
            
            # 1. Fetch metadata directly from LeRobot v3.0 API
            episodes_df = dataset.meta.episodes
            fps = dataset.meta.fps # Get FPS from meta/info.json automatically
            
            pairs = []
            assigned_stems = set()  # Track which video stems have already been matched
            
            # Iterate through each episode defined in LeRobot metadata
            for ep_idx in range(len(episodes_df)):
                ep_meta = episodes_df.iloc[ep_idx] if hasattr(episodes_df, "iloc") else episodes_df[ep_idx]
                
                # Calculate true episode duration in seconds
                episode_len_frames = ep_meta["length"]
                dataset_duration = episode_len_frames / fps
                
                best_match_entry = None
                best_match_stem = None
                best_diff = float("inf")
                
                # 2. Iterate through your raw annotations to find the matching PyAV video duration
                for raw_key, raw_entry in raw.items():
                    # Extract the file stem to look up the PyAV duration
                    # e.g., 'raw_data/make coffee/0cd19f55-cb12-4e3d-9757-0e8f7155b26f.json' -> '0cd19f55-cb12-4e3d-9757-0e8f7155b26f'
                    source_file_path = Path(raw_entry["source_file"])
                    stem = source_file_path.stem
                    
                    # Skip if this specific video stem was already assigned to another parquet episode
                    if stem in assigned_stems:
                        continue
                    
                    # Retrieve the PyAV duration calculated earlier
                    raw_duration = durations_per_episodes.get(stem)
                    if raw_duration is None:
                        # Skip if we don't have an MP4 duration matching this JSON annotation name
                        continue
                    
                    diff = abs(dataset_duration - raw_duration)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_match_entry = raw_entry
                        best_match_stem = stem
                        
                # 3. Pair them if they are within your tolerated duration drift
                if best_match_entry and best_diff <= duration_tolerance:
                    pairs.append((best_match_entry, ep_idx))
                    assigned_stems.add(best_match_stem)  # Lock this raw file so it can't be reused
                    print(
                        f"Matched Episode {ep_idx:>3} (Len: {dataset_duration:.2f}s) "
                        f"to raw file '{best_match_entry.get('source_file')}' (Diff: {best_diff:.2f}s)"
                    )
                else:
                    print(
                        f"WARNING: Could not find a valid matching raw file for Episode {ep_idx} "
                        f"within {duration_tolerance}s tolerance."
                    )
                    
            return pairs

    def _match_by_exact_frame_count(
        self,
        dataset: LeRobotDataset,
        frame_counts_per_episode: dict[str, int],  # {episode_stem: frame_count} (from PyAV)
        raw: dict[str, dict[str, Any]],            # {raw_key: raw_annotation_dict}
    ) -> list[tuple[dict[str, Any], int]]:
        
        episodes_df = dataset.meta.episodes
        num_lerobot = len(episodes_df)
        
        # 1. Group raw items by their exact frame counts to detect identical lengths
        raw_by_frame_count = defaultdict(list)
        for raw_key, raw_entry in raw.items():
            stem = Path(raw_entry["source_file"]).stem
            raw_frames = frame_counts_per_episode.get(stem)
            
            if raw_frames is not None:
                raw_by_frame_count[raw_frames].append((raw_key, raw_entry))

        pairs = []
        mismatched_indices = []
        
        # 2. Match each LeRobot episode to its exact raw frame-count partner
        for ep_idx in range(num_lerobot):
            ep_meta = episodes_df.iloc[ep_idx] if hasattr(episodes_df, "iloc") else episodes_df[ep_idx]
            dataset_frames = int(ep_meta["length"])
            
            candidates = raw_by_frame_count.get(dataset_frames, [])
            
            if not candidates:
                mismatched_indices.append((ep_idx, dataset_frames, "No raw file has this exact frame count."))
                continue
                
            if len(candidates) > 1:
                # Multiple files share the exact same frame count.
                # We must flag this because duration/length alone is ambiguous for these files!
                candidate_names = [Path(c[1]["source_file"]).name for c in candidates]
                mismatched_indices.append(
                    (ep_idx, dataset_frames, f"Ambiguity! Multiple raw files have this frame count: {candidate_names}")
                )
                continue
                
            # Perfect, unique 1-to-1 match
            raw_key, raw_entry = candidates[0]
            pairs.append((raw_entry, ep_idx))
            print(f"  Exact Match: Ep {ep_idx:>3} <--> '{Path(raw_entry['source_file']).stem}' ({dataset_frames} frames)")

        # 3. Handle mismatches and raise an descriptive error if the sync is not perfect
        if mismatched_indices:
            print("\n❌ --- PARQUET SYNC ERRORS FOUND ---")
            for ep_idx, frames, reason in mismatched_indices:
                print(f"  Ep {ep_idx:>3} ({frames} frames) failed: {reason}")
                
            raise ValueError(
                f"Could not automatically recover 1-to-1 mapping. {len(mismatched_indices)} "
                f"out of {num_lerobot} episodes failed the exact-match requirement."
            )

        print(f"\n✅ Perfect sync achieved! All {len(pairs)} episodes aligned with 0-frame drift.")
        return pairs
    
    def _match_with_duration_fallback(
        self,
        dataset: LeRobotDataset,
        frame_counts_per_episode: dict[str, int],  # {episode_stem: frame_count}
        durations_per_episodes: dict[str, float],   # {episode_stem: float_duration}
        raw: dict[str, dict[str, Any]],            # {raw_key: raw_annotation_dict}
        duration_tolerance: float = 0.05,          # Tight tolerance for the fallback check
    ) -> list[tuple[dict[str, Any], EpisodeRecord, int]]:
        
        episodes_df = dataset.meta.episodes
        num_lerobot = len(episodes_df)
        fps = dataset.meta.fps
        
        # 1. Group raw items by their exact frame counts
        raw_by_frame_count = defaultdict(list)
        for raw_key, raw_entry in raw.items():
            stem = Path(raw_entry["source_file"]).stem
            raw_frames = frame_counts_per_episode.get(stem)
            if raw_frames is not None:
                raw_by_frame_count[raw_frames].append((raw_key, raw_entry))

        pairs = []
        assigned_stems = set()  # CRITICAL: Track already-matched raw files!
        for ep_idx in range(num_lerobot):
            ep_meta = episodes_df.iloc[ep_idx] if hasattr(episodes_df, "iloc") else episodes_df[ep_idx]
            dataset_frames = int(ep_meta["length"])
            dataset_duration = dataset_frames / fps
            
            candidates = raw_by_frame_count.get(dataset_frames, [])
            
            # Filter out candidates that have already been matched to an earlier episode
            available_candidates = [
                (k, r) for (k, r) in candidates 
                if Path(r["source_file"]).stem not in assigned_stems
            ]
            
            if not available_candidates:
                print(f"⚠️ Warning: No unassigned raw files match Ep {ep_idx} ({dataset_frames} frames).")
                continue
                
            # Case A: Only one available candidate left (no ambiguity anymore!)
            if len(available_candidates) == 1:
                raw_key, raw_entry = available_candidates[0]
                stem = Path(raw_entry["source_file"]).stem
                pairs.append((raw_entry, episodes_df[ep_idx], ep_idx))
                assigned_stems.add(stem)  # Lock it!
                continue
                
            # Case B: Ambiguity! Multiple unassigned raw files have the same frame count.
            print(f"🔍 Resolving ambiguity for Ep {ep_idx} ({dataset_frames} frames) among {len(available_candidates)} candidates using duration...")
            
            best_candidate = None
            best_stem = None
            best_diff = float("inf")
            
            for raw_key, raw_entry in available_candidates:
                stem = Path(raw_entry["source_file"]).stem
                raw_duration = durations_per_episodes.get(stem)
                
                if raw_duration is None:
                    continue
                    
                diff = abs(dataset_duration - raw_duration)
                if diff < best_diff:
                    best_diff = diff
                    best_candidate = (raw_key, raw_entry)
                    best_stem = stem
            
            if best_candidate and best_diff <= duration_tolerance:
                raw_key, raw_entry = best_candidate
                pairs.append((raw_entry, episodes_df[ep_idx], ep_idx))
                assigned_stems.add(best_stem)  # Lock it!
                print(f"  ✅ Resolved! Ep {ep_idx} matched to '{best_stem}' (Duration Diff: {best_diff:.4f}s)")
            else:
                print(f"  ❌ Failed to resolve Ep {ep_idx} via duration.")
                
        return pairs

    # ------------------------------------------------------------------
    # 4. Stage rows in the exact shape LanguageColumnsWriter expects
    # ------------------------------------------------------------------
    def _stage_episode(self, entry: dict[str, Any], record: EpisodeRecord, staging: EpisodeStaging) -> None:
        rows: list[dict[str, Any]] = []
        print(record)
        for sub in entry["subtasks"]:
            rows.append(
                {
                    "role": self.role,
                    "content": str(sub["transcription"]),
                    "style": self.style,
                    "timestamp": snap_to_frame(float(sub["startTime"]), record.frame_timestamps),
                    "tool_calls": None,
                }
            )
        # "plan" is the staging *module bucket* the writer's persistent-style
        # rows are conventionally read from -- not to be confused with the
        # "plan" language *style* (a different, also-persistent style).
        staging.write("plan", rows)

    # ------------------------------------------------------------------
    # 5. Keep meta/info.json in sync (same as Executor does after writing)
    # ------------------------------------------------------------------
    @staticmethod
    def _sync_info_json(root: Path) -> None:
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            return
        info = load_info(root)
        changed = False

        merged_features = {**info.features, **language_feature_info()}
        if merged_features != info.features:
            info.features = merged_features
            changed = True

        existing = info.tools or []
        names = {(t.get("function") or {}).get("name") for t in existing if isinstance(t, dict)}
        if SAY_TOOL_SCHEMA["function"]["name"] not in names:
            info.tools = [*existing, SAY_TOOL_SCHEMA]
            changed = True

        if changed:
            write_info(info, root)
            print("[annotate] meta/info.json updated with language_persistent/language_events features")

    def _get_frame_counts_of_episodes(self, raw_data_path: Path) -> dict[str, int]:
        per_episode_frames = {}
        
        for subdir in sorted(raw_data_path.iterdir()):
            if not subdir.is_dir():
                continue
                
            for path in subdir.iterdir():
                if not path.is_file() or path.suffix != ".mp4":
                    continue
                
                try:
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        # Get the exact number of frames in the video stream
                        frame_count = stream.frames
                        
                        # Fallback if stream.frames is reporting 0 (some container types do this)
                        if frame_count == 0:
                            frame_count = sum(1 for _ in container.decode(video=0))
                            
                    per_episode_frames[path.stem] = frame_count
                    
                except Exception as e:
                    print(f"Warning: Could not read video {path.name} with PyAV: {e}")
                    continue
                    
        return per_episode_frames

    def _get_durations_of_episodes(
        self, 
        raw_data_path: Path, 
        target_fps: float = 30.0  # Pass your LeRobot dataset's FPS here
    ) -> dict[str, float]:
        per_episode = {}
        
        for subdir in sorted(raw_data_path.iterdir()):
            if not subdir.is_dir():
                continue
                
            for path in subdir.iterdir():
                if not path.is_file() or path.suffix != ".mp4":
                    continue
                
                try:
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        # Get exact frame count
                        frame_count = stream.frames
                        if frame_count == 0:
                            frame_count = sum(1 for _ in container.decode(video=0))
                    
                    # CRITICAL: Calculate duration using the dataset's target FPS,
                    # NOT the video container's physical time_base!
                    normalized_duration = frame_count / target_fps
                    
                    per_episode[path.stem] = normalized_duration
                    
                except Exception as e:
                    print(f"Warning: Could not read video {path.name} with PyAV: {e}")
                    continue
                    
        return per_episode


    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def annotate(
        self,
        dataset_path: Path,
        raw_data_path: Path,
        *,
        duration_tolerance: float = 3.0,
        ambiguity_margin: float = 1.0,
        dry_run: bool = False,
    ) -> None:
        root = Path(dataset_path)
        dataset = LeRobotDataset("local/household3v31", root="household3v31")

        raw_data_path = Path(raw_data_path)
        raw_annotations = self._load_raw_annotations(raw_data_path)

        print("Read the preview below carefully before trusting it.")

        frame_count_per_episodes = self._get_frame_counts_of_episodes(raw_data_path)
        duration_per_episode = self._get_durations_of_episodes(raw_data_path)
        pairs = self._match_with_duration_fallback(dataset, frame_count_per_episodes, duration_per_episode, raw_annotations)

        if dry_run:
            print(f"[annotate] dry-run: would stage+write {len(pairs)} episode(s); stopping here.")
            return

        staging_dir = root / ".annotate_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        records = list(iter_episodes(root))

        for entry, episode_meta, idx in pairs:
            staging = EpisodeStaging(staging_dir, idx)
            self._stage_episode(entry, records[idx], staging)

        # Same validator the real pipeline runs before writing. Event-column
        # checks are all no-ops here since we never stage any events.
        
        validator = StagingValidator(dataset_camera_keys=None)
        report = validator.validate(records, staging_dir)
        for w in report.warnings:
            print(f"[annotate] WARNING: {w}")
        if not report.ok:
            raise RuntimeError("Staging validation failed:\n" + "\n".join(report.errors))
        print(f"[annotate] validator: {report.summary()}")

        writer = LanguageColumnsWriter()
        written = writer.write_all(records, staging_dir, root)
        print(f"[annotate] wrote {len(written)} parquet shard(s)")

        self._sync_info_json(root)
        print("[annotate] done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("raw_data_path", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Preview the episode matching, write nothing.")
    parser.add_argument("--style", default="subtask", help="Persistent language style to tag rows with.")
    parser.add_argument("--role", default="assistant", help="Role to tag rows with.")
    parser.add_argument(
        "--duration-tolerance",
        type=float,
        default=3.0,
        help="(duration-heuristic only) max allowed |episode_duration - max(endTime)| in seconds.",
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=1.0,
        help="(duration-heuristic only) required gap in seconds between best and second-best "
        "candidate before a match is accepted.",
    )
    args = parser.parse_args()

    DatasetAnnotator(style=args.style, role=args.role).annotate(
        args.dataset_path,
        args.raw_data_path,
        duration_tolerance=args.duration_tolerance,
        ambiguity_margin=args.ambiguity_margin,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())