import subprocess
import sys
from pathlib import Path

def sync_video_to_30_fps(video_path: Path, delete_original: bool = False):
    output_path = video_path.parent / f"{video_path.name}.30fps.mp4"
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

def main(dataset: Path):
    for path in dataset.iterdir():
        if path.is_file() and path.suffix == ".mp4":
            sync_video_to_30_fps(path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sync_to_30.py <dataset_path>")
    
    root = Path(sys.argv[1])