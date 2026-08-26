import argparse
from pathlib import Path
from dataset_processor import DatasetProcessor
import sys
import os
import uuid
if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("source", type=str, help="Source dataset location")
    argument_parser.add_argument("destination", type=str, help="Output location")
    argument_parser.add_argument("in_expectation", type=str, help="delphi_flat | delphi_structured | delphi_flat_composite_no_segments")
    argument_parser.add_argument("out_format", type=str, help="lerobot_v21_no_overlay | lerobot_v21_overlay | lerobot_v30_no_overlay | lerobot_v30_overlay | pi_data_sharable")


    expectation_class, out_format_class = None, None
    args = argument_parser.parse_args()
    match args.in_expectation:
        case "delphi_flat":
            from dataset_processor import V3_Delphi27
            expectation_class = V3_Delphi27
        case "delphi_structured":
            from dataset_processor import V3_Delphi27_CompoundTaskFolders
            expectation_class = V3_Delphi27_CompoundTaskFolders
        case "delphi_flat_composite_no_segments":
            from dataset_processor import V3_Delphi27_Composite_NoSegments
            expectation_class = V3_Delphi27_Composite_NoSegments
        case _:
            raise ValueError(f"Unsupported input expectation: {args.in_expectation}")
    match args.out_format:
        case "lerobot_v21_no_overlay":
            from dataset_processor import EgoCentricNoOverlayLeRobot21
            out_format_class = EgoCentricNoOverlayLeRobot21
        case "lerobot_v21_overlay":
            from dataset_processor import EgoCentricUR5OverlayV1LeRobot21
            out_format_class = EgoCentricUR5OverlayV1LeRobot21
        case "lerobot_v21_joints":
            from dataset_processor import LeRobot21Joints
            out_format_class = LeRobot21Joints
        case "lerobot_v21_pose":
            from dataset_processor import LeRobot21Pose
            out_format_class = LeRobot21Pose
        case "lerobot_v30_no_overlay":
            from dataset_processor import EgoCentricNoOverlayLeRobot31
            out_format_class = EgoCentricNoOverlayLeRobot31
        case "lerobot_v30_overlay":
            raise NotImplementedError(f"Unsupported input expectation: {args.out_format}")
        case "pi_data_sharable":
            from dataset_processor import EgoCentricNoOverlayLeRobot31
            out_format_class = EgoCentricNoOverlayLeRobot31
        case _:
            raise ValueError(f"Unsupported input expectation: {args.out_format}")

    # Create dataset processor class and run the conversion
    from convert_dataset import get_episode_paths_as_list

    processor = DatasetProcessor(expectation_class(), out_format_class())
        
    root = Path(args.source)
    output_dir_name = f"tmp/{str(uuid.uuid4())[:6]}"
    repo_id = f"local/{output_dir_name}"
    print(f"{root} => {output_dir_name}")
    
    delphi_episode_paths = get_episode_paths_as_list(root, processor.cfg.use_subdirs_as_compound_tasks)

    for i, episode in enumerate(delphi_episode_paths):
        print(f"  [{i}] '{episode}'")
    
    for delphi_episode_path in delphi_episode_paths:
        if processor.cfg.has_segments:
            delphi_episode = processor.deserialize_into_episode(delphi_episode_path)
            lerobot_dataset = processor.write_to_lerobot_dataset(delphi_episode, repo_id, output_dir_name)
        else:
            episode = processor.deserialize_into_simple_episode(delphi_episode_path)
            lerobot_dataset = processor.write_simple_episode_to_lerobot_dataset(episode, repo_id, output_dir_name)

    if processor.out.version == 31:
        processor.output_dataset.finalize()