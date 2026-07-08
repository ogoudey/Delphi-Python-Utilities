from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import sys

import av
import numpy as np
import pandas as pd
from PIL import Image

@dataclass
class DelphiDataExpectation:
    use_episode_uuid_as_file_stem = False
    pass
"""
    tasks_filename      : str
    joints_filename     : str
    video_filename      : List[str]
 
    # --- tasks.json keys -----------------------------------------------------
    task_start_key      : str
    task_end_key        : str
    task_annotation_key : str
    task_time_format    : str  # float; or "iso8601" if strings
 
    # --- joints.csv columns --------------------------------------------------
    joints_timestamp_col  : str
    joints_timestamp_fmt  : str       # e.g. "2026-03-27T21:16:31.279Z"
    joints_source_col     : str
    joints_joint_col      : str
    joints_position_cols  : List[str]
    joints_rotation_cols  : List[str]
    joints_pinch_col      : str
 
    # --- Segment alignment ---------------------------------------------------
    segment_time_basis    : str      # how start/end times are expressed on Segment
    segment_boundary      : str
"""
class V1(DelphiDataExpectation):
    # --- File names ----------------------------------------------------------
    tasks_filename      = "tasks.json"
    joints_filename     = "data.csv"
    video_filenames      = ["head.mp4"]
 
    # --- tasks.json keys -----------------------------------------------------
    task_start_key      = "startTime"
    task_end_key        = "endTime"
    task_annotation_key = "transcription"
    task_time_format    = "unix_epoch_seconds"   # float; or "iso8601" if strings
 
    # --- joints.csv columns --------------------------------------------------
    joints_timestamp_col  = "Timestamp"
    joints_timestamp_fmt  = "iso8601_utc"        # e.g. "2026-03-27T21:16:31.279Z"
    joints_source_col     = "Source"
    joints_joint_col      = "Joint"
    joints_position_cols  = ["PositionX", "PositionY", "PositionZ"]
    joints_rotation_cols  = ["RotationX", "RotationY", "RotationZ", "RotationW"]
    joints_pinch_col      = "PinchAmount"
 
    # --- Segment alignment ---------------------------------------------------
    segment_time_basis    = "epoch_seconds"      # how start/end times are expressed on Segment
    segment_boundary      = "inclusive" 

class V2_Delphi27(DelphiDataExpectation):
    # --- File names ----------------------------------------------------------
    tasks_filename      = "tasks.json"
    joints_filename     = "state.csv"
    video_filenames      = ["head.mp4"]
 
    # --- tasks.json keys -----------------------------------------------------
    task_start_key      = "startTime"
    task_end_key        = "endTime"
    task_annotation_key = "transcription"
    task_time_format    = "unix_epoch_seconds"   # float; or "iso8601" if strings
 
    # --- joints.csv columns --------------------------------------------------
    joints_timestamp_col  = "Timestamp"
    joints_timestamp_fmt  = "iso8601_utc"        # e.g. "2026-03-27T21:16:31.279Z"
    joints_source_col     = "Source"
    joints_joint_col      = "Joint"
    joints_position_cols  = ["PositionX", "PositionY", "PositionZ"]
    joints_rotation_cols  = ["RotationX", "RotationY", "RotationZ", "RotationW"]
    joints_pinch_col      = "PinchAmount"
 
    # --- Segment alignment ---------------------------------------------------
    segment_time_basis    = "epoch_seconds"      # how start/end times are expressed on Segment
    segment_boundary      = "inclusive"

class V3_Delphi27(DelphiDataExpectation):
    # --- File names ----------------------------------------------------------
    use_episode_uuid_as_file_stem = True
    tasks_filename      = ".json"
    joints_filename     = ".csv"
    video_filenames      = [".mp4"]

    # --- tasks.json keys -----------------------------------------------------
    task_start_key      = "startTime"
    task_end_key        = "endTime"
    task_annotation_key = "transcription"
    task_time_format    = "unix_epoch_seconds"   # float; or "iso8601" if strings
 
    # --- joints.csv columns --------------------------------------------------
    joints_timestamp_col  = "Timestamp"
    joints_timestamp_fmt  = "iso8601_utc"        # e.g. "2026-03-27T21:16:31.279Z"
    joints_source_col     = "Source"
    joints_joint_col      = "Joint"
    joints_position_cols  = ["PositionX", "PositionY", "PositionZ"]
    joints_rotation_cols  = ["RotationX", "RotationY", "RotationZ", "RotationW"]
    joints_pinch_col      = "PinchAmount"
 
    # --- Segment alignment ---------------------------------------------------
    segment_time_basis    = "epoch_seconds"      # how start/end times are expressed on Segment
    segment_boundary      = "inclusive" 

@dataclass
class OutputConfig:
    fps = 30
    merge_handedness_with_joint_name = False # overriden
    pass
"""
    robot_type: str
"""





class EgoCentricUR5OverlayV1LeRobot21(OutputConfig):
    version = 21
    robot_type = "ur5_overlay"

class EgoCentricNoOverlayLeRobot21(OutputConfig):
    version = 21
    robot_type = "human"
    camera_name = "head" # Just one camera - bound to fail later
    merge_handedness_with_joint_name = True
    data_keys_to_exclude = ["base", "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    

class EgoCentricNoOverlayLeRobot31(OutputConfig):
    version = 31
    robot_type = "human"

@dataclass
class TaskAnnotation:
    start_time: float   # seconds from epoch (or relative — whatever the JSON carries)
    end_time: float
    annotation: str

@dataclass
class Episode:
    """All raw data for one recording session."""
    video_paths: List[Path]
    tasks: List[TaskAnnotation]
    state: pd.DataFrame                 # full joint CSV, timestamp-indexed

@dataclass
class Segment:
    """One task's worth of data sliced out of an Episode."""
    video_paths: List[Path]                    # same source file
    video_start_time: float
    video_end_time: float
    task: str
    state: pd.DataFrame                 # joint rows whose timestamp falls in [start, end]



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ISO-8601 timestamps in the CSV look like "2026-03-27T21:16:31.279Z".
# We convert them to seconds-since-epoch (UTC) so they're comparable to the
# float times in tasks.json.

def _iso_to_epoch(ts: str) -> float:
    """'2026-03-27T21:16:31.279Z'  →  float seconds since Unix epoch (UTC)."""
    return pd.Timestamp(ts, tz="UTC").timestamp()


def _parse_task_time(raw: float | str) -> float:
    """
    tasks.json can store times as:
      - a plain float (Unix epoch seconds)
      - an ISO-8601 string
    """
    if isinstance(raw, str): # If it's Unix epoch seconds
        return _iso_to_epoch(raw)
    return float(raw)

def _decode_video_segment(
    video_path: Path,
    start_epoch: float,
    end_epoch: float,
    fps: int,
) -> List[Image.Image]:
    """
    Decode frames from video_path that fall within [start_epoch, end_epoch].
 
    This assumes the video's own start time aligns with the epoch timestamps
    stored in the CSV.
 
    Returns a list of PIL Images sampled at `fps`.
    """
    video_start = 0.0 # Should potentially change
    t0 = max(0.0, start_epoch - video_start)
    t1 = end_epoch - video_start
 
    frames: List[Image.Image] = []
    target_interval = 1.0 / fps
    next_target = t0
 
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"  # seek efficiently
        container.seek(int(t0 * av.time_base ** -1), any_frame=False)  # coarse seek
        stream.codec_context.skip_frame = "DEFAULT"
 
        for packet in container.demux(stream):
            for frame in packet.decode():
                pts_sec = float(frame.pts * stream.time_base)
                if pts_sec < t0:
                    continue
                if pts_sec > t1:
                    return frames
                if pts_sec >= next_target:
                    frames.append(frame.to_image())
                    next_target += target_interval
 
    return frames

def _segment_video_by_task(episode: Episode) -> List[Segment]:
        """
        Slice an Episode into one Segment per TaskAnnotation.

        Each Segment carries:
        - the path to the source video (trimming happens in Part II)
        - the wall-clock window [video_start_time, video_end_time]
        - the subset of joint rows whose epoch_time falls within that window
        - the single task annotation string
        """
        segments: List[Segment] = []

        for task in episode.tasks:
            mask = (
                (episode.state["epoch_time"] >= task.start_time) &
                (episode.state["epoch_time"] <= task.end_time)
            )
            state_slice = episode.state.loc[mask].copy().reset_index(drop=True)

            segments.append(
                Segment(
                    video_paths=episode.video_paths,
                    video_start_time=task.start_time,
                    video_end_time=task.end_time,
                    task=task.annotation,
                    state=state_slice,
                )
            )

        return segments


# Processor class

class DatasetProcessor:
    cfg: DelphiDataExpectation
    out: OutputConfig
    def __init__(self, dde: DelphiDataExpectation, out: OutputConfig):
        self.cfg = dde
        self.out = out
        
        self.output_dataset = None

    def deserialize(self, episode_path: Path) -> Episode:
        episode_path = Path(episode_path)

        # --- tasks.json ----------------------------------------------------------
        tasks_file = episode_path.with_suffix(self.cfg.tasks_filename) if self.cfg.use_episode_uuid_as_file_stem else episode_path / self.cfg.tasks_filename 
        with tasks_file.open() as f:
            raw_tasks = json.load(f)

        tasks: List[TaskAnnotation] = [
            TaskAnnotation(
                start_time=_parse_task_time(t[self.cfg.task_start_key]),
                end_time=_parse_task_time(t[self.cfg.task_end_key]),
                annotation=t[self.cfg.task_annotation_key],
            )
            for t in raw_tasks
        ]

        joints_file = episode_path.with_suffix(self.cfg.joints_filename) if self.cfg.use_episode_uuid_as_file_stem else episode_path / self.cfg.joints_filename
        state = pd.read_csv(joints_file)
        state.columns = state.columns.str.strip()
        state["epoch_time"] = state[self.cfg.joints_timestamp_col].apply(_iso_to_epoch)
        state = state.sort_values("epoch_time").reset_index(drop=True)

        # Normalize to relative time: first joint row = t=0.0
        state["epoch_time"] -= state["epoch_time"].iloc[0]

        # Exclude undesired data
        if self.out.data_keys_to_exclude:
            state = state[~state[self.cfg.joints_joint_col].isin(self.out.data_keys_to_exclude)]
        
        
        
        if self.out.merge_handedness_with_joint_name:
            # e.g. source="right", joint="Wrist" -> joint="RightWrist"

            source = state[self.cfg.joints_source_col].str.strip().str.lower()
            is_handed = source.isin(["right", "left"])

            state[self.cfg.joints_joint_col] = np.where(
                is_handed,
                source.str.capitalize() + state[self.cfg.joints_joint_col].str.strip(),
                state[self.cfg.joints_joint_col]
            )
        
        # Pivot: one row per timestamp, one column per (joint, field)
        state = state.pivot_table(
            index="epoch_time",
            columns=self.cfg.joints_joint_col,
            values=self.cfg.joints_position_cols + self.cfg.joints_rotation_cols + [self.cfg.joints_pinch_col],
            aggfunc="first"
        )

        # Flatten MultiIndex columns: ("PositionX", "shoulder_pan_joint") → "shoulder_pan_joint/PositionX"
        state.columns = [f"{joint}/{field}" for field, joint in state.columns]
        state = state.sort_index().reset_index()

        # TODO: At some point in this, I'd like to not output resulting rows with these joints: `data_keys_to_exclude`. As in, that data's no good - remove them.


        video_paths = []        
        for video_filename in self.cfg.video_filenames:
            video_path = episode_path.with_suffix(video_filename) if self.cfg.use_episode_uuid_as_file_stem else episode_path / video_filename
            if not video_path.exists():
                raise FileNotFoundError(f"Expected video at {video_path}")
            video_paths.append(video_path)
        return Episode(
            video_paths=video_paths,
            tasks=tasks,
            state=state,
        )


    

    def create_lerobot_dataset(self, representative_segment: Segment, repo_id: str, output_path: Path):
        sys.path.append("/home/olin/Robotics/Projects/lerobot") # This links to a lerobot package, which is often local
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        match self.out.version:
            case 21:

                # Build the features schema from the first segment's state columns
                state_cols = [c for c in representative_segment.state.columns if c != "epoch_time"]
                n_state = len(state_cols)
        
                camera_names = [self.out.camera_name]#[p.stem for p in representative_segment.video_paths]
                
                features = {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": (n_state,),
                        "names": {"joints": state_cols},
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": (n_state,),
                        "names": {"joints": state_cols},
                    },
                    **{
                        f"observation.images.{cam}": {
                            "dtype": "video",
                            "shape": (480, 640, 3),
                            "names": ["height", "width", "channel"],
                        }
                        for cam in camera_names
                    },
                }
                self.output_dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    root=output_path,
                    fps=self.out.fps,
                    robot_type=self.out.robot_type,
                    features=features,
                    use_videos=True,
                )
            case 31:
                pass
            case _:
                raise ValueError("Please specify in the output config a LeRobotDataset version (e.g 31 for v3.1). Supported: v3.1 and v2.1")

    def write_to_lerobot_dataset(self,
        episode: Episode,
        repo_id: str,
        output_path: str | Path

    ):
        """
        Write a list of Segments to a LeRobotDataset on disk.
 
        Each Segment becomes one LeRobot episode.  Video frames are decoded
        from the source .mp4 via PyAV, trimmed to [video_start_time,
        video_end_time], and re-encoded by LeRobot.
 
        Args:
            segments:     List of segments
            repo_id:      HuggingFace-style repo id, e.g. "local/datasetname", "labyrinthai/ourfirstupload"
            output_path:  local root directory for the dataset
            fps:          target frame-rate for the LeRobot dataset (30)
 
        Returns:
            LeRobotDataset (finalised, ready for push_to_hub or local use)
        """
        
 
        output_path = Path(output_path)
 
        match self.out.version:
            case 21:
                segments = _segment_video_by_task(episode)

                if not self.output_dataset:
                    self.create_lerobot_dataset(representative_segment=segments[0], repo_id=repo_id, output_path=output_path) 
        
                for seg in segments:
                    print(f"task: {seg.task}")
                    print(f"  window:     {seg.video_start_time:.3f} → {seg.video_end_time:.3f}")
                    print(f"  state rows: {len(seg.state)}")
                    if len(seg.state):
                        print(f"  state time: {seg.state['epoch_time'].min():.3f} → {seg.state['epoch_time'].max():.3f}")

                for seg in segments:
                    # Decode all cameras; key by camera name derived from filename stem
                    all_frames = {
                        self.out.camera_name: _decode_video_segment( # key might want to be video_path.stem !! assumes one video!
                            video_path,
                            seg.video_start_time,
                            seg.video_end_time,
                            self.out.fps,
                        )
                        for video_path in seg.video_paths
                    }

                    # Trim to shortest camera (guards against minor length mismatches)
                    n_frames = min(len(f) for f in all_frames.values())
                    all_frames = {k: v[:n_frames] for k, v in all_frames.items()}
                    
                    state_cols = [c for c in segments[0].state.columns if c != "epoch_time"]
                    state_matrix = seg.state[state_cols].to_numpy(dtype=np.float32)
                    seg_epoch_times = seg.state["epoch_time"].to_numpy()
                    frame_times = seg.video_start_time + np.arange(n_frames) / self.out.fps

                    for i, t in enumerate(frame_times):
                        row_idx = int(np.argmin(np.abs(seg_epoch_times - t)))
                        state_vec = state_matrix[row_idx]

                        self.output_dataset.add_frame(
                            {
                                "observation.state": state_vec,
                                "action": state_vec,
                                **{f"observation.images.{cam}": frames[i] for cam, frames in all_frames.items()},
                            },
                            task=seg.task,
                        )

                    self.output_dataset.save_episode()
                print(f"Produced {len(segments)} segment(s):\n")
                for i, seg in enumerate(segments):
                    print(
                        f"  [{i}] '{seg.task}'\n"
                        f"       time  : {seg.video_start_time:.3f} → {seg.video_end_time:.3f} s\n"
                        f"       joints: {len(seg.state)} rows\n")
            case 31:
                self.output_dataset.finalize() # v3.1 thing...
            case _:
                raise ValueError("Please specify in the output config a LeRobotDataset version (e.g 31 for v3.1). Supported: v3.1 and v2.1")
        return

# main tests one episode
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python3 dataset_processor.py <episode_directory> <repo_id> <output_path>")
        sys.exit(1)
    
    # Processor definition
    processor = EpisodeProcessor(V3_Delphi27(), EgoCentricNoOverlayLeRobot21())
    delphi_episode: Episode = processor.deserialize(Path(sys.argv[1]))
    lerobot_dataset = processor.write_to_lerobot_dataset(delphi_episode, sys.argv[2], sys.argv[3])
