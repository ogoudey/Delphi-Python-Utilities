import subprocess
import sys
from pathlib import Path

def sync_video_to_30_fps(video_path: Path, delete_original: bool = False):
    output_path = video_path.with_suffix(".30fps.mp4")
    subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-vsync", "cfr",
        "-r", "30",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "copy",
        f"{output_path}"
    ], check=True)
    if delete_original:
        video_path.unlink()

def main(dataset: Path, delete_original: bool = False):
    cnt = 0
    for path in dataset.iterdir():
        if path.is_file() and path.suffix == ".mp4" and not path.name.endswith(".30fps.mp4"):
            cnt += 1
            sync_video_to_30_fps(path, delete_original)
    print(f"Synced {cnt} videos to 30 fps in {dataset}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sync_to_30.py <dataset_path> <bool: delete_original = false>")
        sys.exit(1)

    root = Path(sys.argv[1])
    if len(sys.argv) > 2:
        main(root, delete_original=sys.argv[2].lower() == "true")
    else:
        main(root)

"""
Use in WF:

```
python3 sync_to_30.py tmp/data true
```


"""