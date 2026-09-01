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
    use_subdirs_as_compound_tasks = False
    has_segments = True
    in_local_quaternions = True
    
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

class V3_Delphi27_CompoundTaskFolders(DelphiDataExpectation):
    # --- File names ----------------------------------------------------------
    use_subdirs_as_compound_tasks = True
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

class V3_Delphi27_Composite_NoSegments(DelphiDataExpectation):
    has_segments = False
    use_episode_uuid_as_file_stem = True
    episode_metadata = ".episode_metadata.json"
    state_filename     = ".robot_posture_1.csv"
    video_filenames      = [".camera_1.mp4"]
    joints_timestamp_col  = "Timestamp"
    joints_source_col     = "Source"
    joints_joint_col      = "Joint"
    joints_position_cols  = ["PositionX", "PositionY", "PositionZ"]
    joints_rotation_cols  = ["RotationX", "RotationY", "RotationZ", "RotationW"]
    joints_pinch_col      = "PinchAmount"
    end_effector_name = "gripper"

@dataclass
class OutputConfig:
    version = None
    fps = 30
    merge_handedness_with_joint_name = False # overriden
    use_first_person = True
    proprioception_mode = None
    write_modality_json = False
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
    
    #data_keys_to_exclude = ["base", "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

class LeRobot21Pose(OutputConfig):
    fps = 30
    version = 21
    robot_type = "rendered_ur5"
    camera_name = "head" # Just one camera - bound to fail later
    merge_handedness_with_joint_name = True
    proprioception_mode = "target_pose"
    data_keys_to_exclude = ["base", "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"] 
    target_name = "endEffector"
    write_modality_json = True

class LeRobot21Joints(OutputConfig):
    fps = 30
    version = 21
    robot_type = "rendered_ur5"
    camera_name = "head" # Just one camera - bound to fail later
    merge_handedness_with_joint_name = True
    proprioception_mode = "joints"
    data_keys_to_exclude = ["base", "endEffector"] 
    write_modality_json = True
class EgoCentricNoOverlayLeRobot31(OutputConfig):
    version = 31
    robot_type = "human"
    camera_name = "head"
    use_first_person = False
    merge_handedness_with_joint_name = True
    data_keys_to_exclude = ["base", "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]



@dataclass
class TaskAnnotation:
    start_time: float   # seconds from epoch (or relative — whatever the JSON carries)
    end_time: float
    annotation: str

@dataclass
class Episode:
    """All raw data for one recording session. Output of Delphi27 segmentation-via-transcription."""
    video_paths: List[Path]
    tasks: List[TaskAnnotation]
    state: pd.DataFrame                 # full joint CSV, timestamp-indexed
    compound_task_annotation: str = ""


@dataclass
class SimpleEpisode:
    """A simplified version of an episode with basic data."""
    video_paths: List[Path]
    state: pd.DataFrame                 # full joint CSV, timestamp-indexed
    task: str

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

def _video_shape(video_path: Path) -> tuple:
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        return (stream.height, stream.width, 3)

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

def _decode_video_segment_generator(
    video_path: Path,
    start_epoch: float,
    end_epoch: float,
    fps: int,
):
    t0 = max(0.0, start_epoch)
    t1 = end_epoch
    target_interval = 1.0 / fps
    next_target = t0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        container.seek(int(t0 / float(stream.time_base)), any_frame=False)
        stream.codec_context.skip_frame = "DEFAULT"

        for packet in container.demux(stream):
            for frame in packet.decode():
                pts_sec = float(frame.pts * stream.time_base)
                if pts_sec < t0:
                    continue
                if pts_sec > t1:
                    return
                if pts_sec >= next_target:
                    yield frame.to_image()
                    next_target += target_interval

def _get_language_persistent_from_episode(episode: Episode) -> list:
    # compound task covers the whole episode starting at 0.0
    atoms = [
        {
            "role": "assistant",
            "content": episode.compound_task_annotation,
            "style": "subtask",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        }
    ]
    # individual subtasks fire at their start_time
    for task in episode.tasks:
        atoms.append({
            "role": "assistant",
            "content": task.annotation,
            "style": "subtask",
            "timestamp": task.start_time,
            "camera": None,
            "tool_calls": None,
        })
    return atoms

def _get_episode_duration(episode: Episode | SimpleEpisode) -> float:
    return float(episode.state["epoch_time"].iloc[-1])

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
                    task=task.annotation, # here is where we'd want to edit the annotation
                    state=state_slice,
                )
            )

        return segments


# Processor class

class DatasetProcessor:
    cfg: DelphiDataExpectation
    out: OutputConfig
    MY_TID2ANNOT8_MAP = {
        "SortUtensils": "touch the orange"
    }

    def __init__(self, dde: DelphiDataExpectation, out: OutputConfig):
        self.cfg = dde
        self.out = out
        
        self.output_dataset = None
        self.local_quaternion_to_joint_value_map = {}

    def deserialize_into_episode(self, episode_path: Path) -> Episode:
        episode_path = Path(episode_path)

        # --- compound task annotation ---
        print(episode_path)
        print(episode_path.parts)
        compound_task_annotation = str(episode_path.parts[-2]) if self.cfg.use_subdirs_as_compound_tasks else "untracked"

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
            aggfunc="first",
        )

        # Flatten MultiIndex columns: ("PositionX", "shoulder_pan_joint") → "shoulder_pan_joint/PositionX"
        state.columns = [f"{joint}/{field}" for field, joint in state.columns]
        state = state.sort_index().reset_index()

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
            compound_task_annotation=compound_task_annotation
        )

    def deserialize_into_simple_episode(self, episode_path: Path) -> SimpleEpisode:
        # ---------------- metadata -----------------
        episode_metadata_file = episode_path.with_suffix(self.cfg.episode_metadata) if self.cfg.use_episode_uuid_as_file_stem else episode_path / self.cfg.episode_metadata
        with episode_metadata_file.open() as f:
            episode_metadata = json.load(f)
            try:
                tid = episode_metadata["content"]["episodeTaskId"]
                language = DatasetProcessor.MY_TID2ANNOT8_MAP[tid]
            except KeyError as e:
                raise ValueError(f"Missing {e}. {DatasetProcessor.MY_TID2ANNOT8_MAP}")
            print(f"--------------start metadata----------------\n{episode_metadata}\n--------------end metadata----------------")

        state_file = episode_path.with_suffix(self.cfg.state_filename)
        state = pd.read_csv(state_file)
        state.columns = state.columns.str.strip()  # moved above the calibration call

        # Standardize time
        state["epoch_time"] = state[self.cfg.joints_timestamp_col].apply(_iso_to_epoch)
        state = state.sort_values("epoch_time").reset_index(drop=True)
        state["epoch_time"] -= state["epoch_time"].iloc[0]

        # Pivot
        value_cols = self.cfg.joints_position_cols + self.cfg.joints_rotation_cols + [self.cfg.joints_pinch_col]
        long = state.set_index(["epoch_time", self.cfg.joints_joint_col])[value_cols]
        state = long.groupby(level=[0, 1]).first().unstack(level=1)

        # ---------------- state ------------------
        if self.out.proprioception_mode == "joints":
            if self.cfg.in_local_quaternions and not self.local_quaternion_to_joint_value_map:
                self._calculate_joint_value_map(state)

            arm_joints = [
                j for j in pd.unique(state[self.cfg.joints_joint_col])
                if j not in self.out.data_keys_to_exclude and j != self.cfg.end_effector_name
            ]

            

            

            # ---- apply joint_name -> joint_value ----
            rx, ry, rz, rw = self.cfg.joints_rotation_cols

            angle_cols = {}
            for joint in arm_joints:
                axis_col = self.local_quaternion_to_joint_value_map.get(joint)
                if axis_col is None:
                    raise KeyError(
                        f"joint '{joint}' has no resolved axis in local_quaternion_to_joint_value_map — "
                        f"this episode didn't have enough motion to calibrate it, and no earlier "
                        f"episode did either. Process a higher-motion episode for this joint first."
                    )
                component = state[(axis_col, joint)]
                w = state[(rw, joint)]
                angle_cols[joint] = np.unwrap(2.0 * np.arctan2(component, w))

            # gripper + final assembly — once, outside the loop
            gripper_col = ("PinchAmount", self.cfg.end_effector_name)
            if gripper_col not in state.columns:
                raise KeyError(
                    f"expected pinch column {gripper_col} not found — check that "
                    f"'{self.cfg.end_effector_name}' is the exact joint-name value in "
                    f"{self.cfg.joints_joint_col} and 'PinchAmount' is in the values list "
                    f"passed to pivot_table."
                )
            gripper_series = state[gripper_col]

            state = pd.DataFrame({**angle_cols, "gripper": gripper_series}, index=state.index)
            state = state.sort_index().reset_index()
        elif self.out.proprioception_mode == "target_pose":
            target_pos_cols = self.cfg.joints_position_cols   # e.g. ["PositionX", "PositionY", "PositionZ"]
            target_rot_cols = self.cfg.joints_rotation_cols   # e.g. ["RotationX", "RotationY", "RotationZ", "RotationW"]

            target_cols = {}
            for col in target_pos_cols + target_rot_cols:
                key = (col, self.out.target_name)
                if key not in state.columns:
                    raise KeyError(
                        f"expected target column {key} not found — check that "
                        f"'{self.out.target_name}' is the exact row value in "
                        f"{self.cfg.joints_joint_col}, and that {col} is in the "
                        f"values list passed to pivot_table."
                    )
                target_cols[f"{col}"] = state[key]

            gripper_col = ("PinchAmount", self.cfg.end_effector_name)
            if gripper_col not in state.columns:
                raise KeyError(
                    f"expected pinch column {gripper_col} not found — check that "
                    f"'{self.cfg.end_effector_name}' is the exact joint-name value in "
                    f"{self.cfg.joints_joint_col} and 'PinchAmount' is in the values list "
                    f"passed to pivot_table."
                )
            gripper_series = state[gripper_col]

            state = pd.DataFrame({**target_cols, "gripper": gripper_series}, index=state.index)
            state = state.sort_index().reset_index()
        

        # ------------ videos -----------------
        video_paths = []        
        for video_filename in self.cfg.video_filenames:
            video_path = episode_path.with_suffix(video_filename) if self.cfg.use_episode_uuid_as_file_stem else episode_path / video_filename
            if not video_path.exists():
                raise FileNotFoundError(f"Expected video at {video_path}")
            video_paths.append(video_path)

        return SimpleEpisode(
            video_paths=video_paths,
            state=state,
            task=language
        )

    def _calculate_joint_value_map(self, state):
        """
        Populate self.local_quaternion_to_joint_value_map: {joint_name: axis_col},
        where axis_col is whichever of X/Y/Z carries this joint's hinge motion,
        determined empirically from this episode's long-form data (one row per
        joint per timestamp). W is never a candidate — it varies with any
        rotation regardless of axis, so it carries no axis information.

        Only resolves joints with enough motion in THIS episode to tell;
        leaves the rest unresolved (missing from the dict) so a later,
        higher-motion episode can fill them in via the same not-truthy check
        upstream — though note that check is `not self.local_quaternion_to_joint_value_map`,
        i.e. "dict is empty", so it'll only fire again on the very first
        episode if the dict starts non-empty. If you expect calibration to
        need multiple episodes, that condition should probably be "any
        intake joint missing from the map" rather than "map is falsy" —
        worth revisiting once you see whether one episode is ever enough.
        """
        joint_col = self.cfg.joints_joint_col
        rx, ry, rz, rw = self.cfg.joints_rotation_cols
        axis_candidates = [rx, ry, rz]
        tol = getattr(self.cfg, "joint_axis_tol", 1e-3)
        min_std = getattr(self.cfg, "joint_axis_min_std", 1e-2)

        for joint, group in state.groupby(joint_col):
            if joint in self.out.data_keys_to_exclude:
                continue
            if joint in self.local_quaternion_to_joint_value_map:
                continue

            stds = {c: group[c].std() for c in axis_candidates}
            axis_col = max(stds, key=stds.get)
            if stds[axis_col] < min_std:
                continue  # not enough motion this episode to tell yet

            for c in axis_candidates:
                if c == axis_col:
                    continue
                max_abs = group[c].abs().max()
                if max_abs > tol:
                    raise ValueError(
                        f"joint '{joint}': off-axis component {c} peaks at {max_abs:.4f} "
                        f"(> tol={tol}) while candidate hinge axis {axis_col} has "
                        f"std={stds[axis_col]:.4f}. Not a clean single-axis hinge — inspect."
                    )

            self.local_quaternion_to_joint_value_map[joint] = axis_col
    
    

    def create_lerobot_dataset(self, representative_segment: Segment | Episode | SimpleEpisode, repo_id: str, output_path: Path):
        sys.path.append("/home/olin/Robotics/Labyrinth/lerobot") # This links to a lerobot package, which is often local
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
                            "shape": _video_shape(representative_segment.video_paths[0]), # assumes only one camera again
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
                if self.out.write_modality_json:
                    modality_path = output_path / "meta" / "modality.json"
                    configuration_path = output_path / "meta" / "python" / "rendered_ur5_config.py"
                    if not modality_path.exists():
                        import groot_stuff
                        groot_stuff._write_modality_json_and_config(
                            modality_path=modality_path,
                            config_path=configuration_path,
                            embodiment_name="rendered_ur5",
                            joints=list(range(8)),  # -> n_rotation_dims = 7
                            n_pinch=1,
                            video_key="head",
                            joint_key_name="target_pose",
                            gripper_key_name="gripper",
                            joint_action_type="EEF",  # <-- confirm this against your actual data
                        )


                        #groot_stuff.write_modality_json(
                        #    modality_path=modality_path,
                        #    joints=state_cols,
                        #    n_pinch=1,
                        #    video_key=self.out.camera_name,
                        #    task=representative_segment.task
                        #)
            case 31:
                from lerobot.datasets.language import language_feature_info
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
                            "shape": _video_shape(representative_segment.video_paths[0]), # assumes only one camera again
                            "names": ["height", "width", "channel"],
                        }
                        for cam in camera_names
                    },
                    #**language_feature_info(),
                }
                self.output_dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    root=output_path,
                    fps=self.out.fps,
                    robot_type=self.out.robot_type,
                    features=features
                )
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
 
        Returns:
            LeRobotDataset (finalized, ready for push_to_hub or local use)
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

                for i, seg in enumerate(segments):
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

                    # First pass: resolve the state vector for every frame in this segment.
                    
                    frame_states = []
                    for t in frame_times:
                        row_idx = int(np.argmin(np.abs(seg_epoch_times - t)))
                        frame_states.append(state_matrix[row_idx])

                    # Second pass: essentially action[t] = state[t+1], last frame pads with its own state.
                    print(f"Segment {i}/{len(segments)} \"{seg.task}\". Adding {n_frames} frames...")
                    for j in range(n_frames):
                        state_vec = frame_states[j]
                        action_vec = frame_states[j + 1] if j + 1 < n_frames else frame_states[j]

                        self.output_dataset.add_frame(
                            {
                                "observation.state": state_vec,
                                "action": action_vec,
                                **{f"observation.images.{cam}": frames[j] for cam, frames in all_frames.items()},
                            },
                            task=seg.task,
                        )
                    print(f"Segment {i}/{len(segments)} \"{seg.task}\". Saving segment => episode...")

                print(f"Produced {len(segments)} segment(s):\n")
                for i, seg in enumerate(segments):
                    print(
                        f"  [{i}] '{seg.task}'\n"
                        f"       time  : {seg.video_start_time:.3f} → {seg.video_end_time:.3f} s\n"
                        f"       joints: {len(seg.state)} rows\n")
            case 31:
                if not self.output_dataset:
                    self.create_lerobot_dataset(representative_segment=episode, repo_id=repo_id, output_path=output_path) 

                state_cols = [c for c in episode.state.columns if c != "epoch_time"]
                state_matrix = episode.state[state_cols].to_numpy(dtype=np.float32)
                seg_epoch_times = episode.state["epoch_time"].to_numpy()
                duration = _get_episode_duration(episode)

                generators = {
                    self.out.camera_name: _decode_video_segment_generator(
                        video_path, 0.0, duration, self.out.fps,
                    )
                    for video_path in episode.video_paths
                }

                prev_state_vec = None
                prev_cam_dict = None

                for frame_idx, cam_frames in enumerate(zip(*generators.values())):
                    cam_dict = dict(zip(generators.keys(), cam_frames))
                    t = frame_idx / self.out.fps
                    row_idx = int(np.argmin(np.abs(seg_epoch_times - t)))
                    state_vec = state_matrix[row_idx]

                    if prev_state_vec is not None:
                        self.output_dataset.add_frame({
                            "observation.state": prev_state_vec,
                            "action": state_vec,              # action[t] = state[t+1]
                            **{f"observation.images.{cam}": img for cam, img in prev_cam_dict.items()},
                            #"language_persistent": [],
                            #"language_events": [],
                            "task": episode.compound_task_annotation,
                        })

                    prev_state_vec = state_vec
                    prev_cam_dict = cam_dict

                # last frame: action pads with own state
                if prev_state_vec is not None:
                    self.output_dataset.add_frame({
                        "observation.state": prev_state_vec,
                        "action": prev_state_vec,
                        **{f"observation.images.{cam}": img for cam, img in prev_cam_dict.items()},
                        #"language_persistent": None,
                        #"language_events": None,
                        "task": episode.compound_task_annotation,
                    })

                print(f"Saving episode '{episode.compound_task_annotation}'.")
                self.output_dataset.save_episode()
            case _:
                raise ValueError("Please specify in the output config a LeRobotDataset version (e.g 31 for v3.1). Supported: v3.1 and v2.1")
        return
    
    def write_simple_episode_to_lerobot_dataset(self,
        simple_episode: SimpleEpisode,
        repo_id: str,
        output_path: str | Path

    ):  
        output_path = Path(output_path)

        if not self.output_dataset:
            self.create_lerobot_dataset(representative_segment=simple_episode, repo_id=repo_id, output_path=output_path) 

        state_cols = [c for c in simple_episode.state.columns if c != "epoch_time"]
        state_matrix = simple_episode.state[state_cols].to_numpy(dtype=np.float32)
        seg_epoch_times = simple_episode.state["epoch_time"].to_numpy()
        duration = _get_episode_duration(simple_episode)

        generators = {
            self.out.camera_name: _decode_video_segment_generator(
                video_path, 0.0, duration, self.out.fps,
            )
            for video_path in simple_episode.video_paths
        }

        prev_state_vec = None
        prev_cam_dict = None

        for frame_idx, cam_frames in enumerate(zip(*generators.values())):
            cam_dict = dict(zip(generators.keys(), cam_frames))
            t = frame_idx / self.out.fps
            row_idx = int(np.argmin(np.abs(seg_epoch_times - t)))
            state_vec = state_matrix[row_idx]

            if prev_state_vec is not None:
                self.output_dataset.add_frame({
                    "observation.state": prev_state_vec,
                    "action": state_vec,              # action[t] = state[t+1]
                    **{f"observation.images.{cam}": img for cam, img in prev_cam_dict.items()}
                }, task=simple_episode.task)
            prev_state_vec = state_vec
            prev_cam_dict = cam_dict

        if prev_state_vec is not None:
            self.output_dataset.add_frame({
                "observation.state": prev_state_vec,
                "action": prev_state_vec,
                **{f"observation.images.{cam}": img for cam, img in prev_cam_dict.items()}
            }, task=simple_episode.task)

        print(f"Saving episode '{simple_episode.task}'.")
        self.output_dataset.save_episode()
            
# main tests one episode
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python3 dataset_processor.py <episode_directory> <repo_id> <output_path>")
        sys.exit(1)
    
    # Processor definition
    processor = DatasetProcessor(V3_Delphi27_CompoundTaskFolders(), EgoCentricNoOverlayLeRobot31())
    delphi_episode: Episode = processor.deserialize(Path(sys.argv[1]))
    lerobot_dataset = processor.write_to_lerobot_dataset(delphi_episode, sys.argv[2], sys.argv[3])
    processor.output_dataset.finalize() # v3.1 thing...
