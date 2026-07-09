## DelphiToLeRobot

### Delphi => LeRobotDataset

Change `sys.path.append("/home/...)` to your LeRobot install.

Turns a dataset downloaded from Delphi (with given version) to a LeRobotDataset of desired format (change `DatasetProcessor` constructor to change the format).
```
Usage: python3 convert_dataset.py <input_dataset_path> <repo_id> <output_dataset_path>
```

### Compositor
Composites a raw video with a robot overlay, downloaded from delphi.
```
Usage: python3 composite.py <raw_video> <robot_overlay>
```

