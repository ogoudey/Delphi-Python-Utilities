from dataset_processor import DatasetProcessor, V3_Delphi27, EgoCentricNoOverlayLeRobot21
import sys
from pathlib import Path

<<<<<<< HEAD

def get_episode_paths_as_list(dataset_root: Path, use_subdirs_as_compound_tasks: bool):
    delphi_episode_paths = set()
    if use_subdirs_as_compound_tasks:
=======
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 convert_dataset.py <dataset_path> <repo_id> <output_path>")
    processor = DatasetProcessor(V3_Delphi27(), EgoCentricNoOverlayLeRobot21())
    
    root = Path(sys.argv[1])
    delphi_episode_paths = []
    print("Traversing input data...")
    if processor.cfg.use_subdirs_as_compound_tasks:
>>>>>>> 089a2eec70362bb9d66faaf3948cad6cf725ab3b
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
<<<<<<< HEAD
            delphi_episode_paths.add(path)
    return delphi_episode_paths

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 convert_dataset.py <dataset_path> <repo_id> <output_path>")
    processor = DatasetProcessor(V3_Delphi27_CompoundTaskFolders(), EgoCentricNoOverlayLeRobot31())
    
    root = Path(sys.argv[1])
    
    delphi_episode_paths = get_episode_paths_as_list(root, processor.cfg.use_subdirs_as_compound_tasks)

=======
            delphi_episode_paths.append(path)
    
>>>>>>> 089a2eec70362bb9d66faaf3948cad6cf725ab3b
    for i, episode in enumerate(delphi_episode_paths):
        print(f"  [{i}] '{episode}'")
    
    for i, delphi_episode_path in enumerate(delphi_episode_paths):
        print(f"{i}/{len(delphi_episode_paths)} Processing {delphi_episode_path.name}")
        delphi_episode = processor.deserialize(delphi_episode_path)
        print(f"{i}/{len(delphi_episode_paths)} {delphi_episode_path.name} deserialized. Writing...")
        lerobot_dataset = processor.write_to_lerobot_dataset(delphi_episode, sys.argv[2], sys.argv[3])
<<<<<<< HEAD

=======
        print(f"{i}/{len(delphi_episode_paths)} {delphi_episode_path.name} written.")
>>>>>>> 089a2eec70362bb9d66faaf3948cad6cf725ab3b
    if processor.out.version == 31:
        processor.output_dataset.finalize()
