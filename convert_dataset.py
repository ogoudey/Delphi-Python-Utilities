from dataset_processor import DatasetProcessor, V3_Delphi27, EgoCentricNoOverlayLeRobot21
import sys
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 convert_dataset.py <dataset_path> <repo_id> <output_path>")
    processor = DatasetProcessor(V3_Delphi27(), EgoCentricNoOverlayLeRobot21())
    
    root = Path(sys.argv[1])
    delphi_episode_paths = []

    if processor.cfg.use_subdirs_as_compound_tasks:
        for subdir in root.iterdir():
            for path in subdir.iterdir():
                if not path.is_file():
                    continue
                path = path.with_suffix("")
                if path in delphi_episode_paths:
                    continue
                delphi_episode_paths.append(path)
    else:
        for path in root.iterdir():
            if not path.is_file():
                continue
            path = path.with_suffix("")
            if path in delphi_episode_paths:
                continue
            delphi_episode_paths.append(path)

    for i, episode in enumerate(delphi_episode_paths):
        print(f"  [{i}] '{episode}'")
    
    for delphi_episode_path in delphi_episode_paths:
        print(f"Processing {delphi_episode_path.name}")
        continue
        delphi_episode = processor.deserialize(delphi_episode_path)
        lerobot_dataset = processor.write_to_lerobot_dataset(delphi_episode, sys.argv[2], sys.argv[3])
    if processor.out.version == 31:
        processor.output_dataset.finalize()
