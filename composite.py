import subprocess
import sys
from pathlib import Path

def composite(raw_video: Path, overlay_video: Path, output_path: Path, delete_originals: bool = False):
    subprocess.run([
        "ffmpeg",
        "-i", str(raw_video),       # background (raw video)
        "-i", str(overlay_video),   # foreground (robot on red background)
        "-filter_complex",
        "[1:v]colorkey=0xFF0000:0.3:0.1[fg];[0:v][fg]overlay",
        "-c:v", "libx264",
        str(output_path),
    ], check=True)
    if delete_originals:
        raw_video.unlink()
        overlay_video.unlink()


def main(dataset: Path, delete_originals: bool = False):
    cnt = 0
    for path in dataset.iterdir():
        print(f"Processing {path.name}")
        if not "camera_1" in path.name:
            continue
        overlay_path = path.parent / path.name.replace("camera_1", "overlay_1")
        if not overlay_path.exists():
            print(f"Overlay video not found for {path.name}. {path.name.replace('camera_1', 'overlay_1')} not found.")
            continue

        output_path = path.with_suffix(".composite.mp4")
        if path.is_file() and path.suffix == ".mp4":
            cnt += 1
            composite(path, overlay_path, output_path, delete_originals)
    print(f"Composite {cnt} videos in {dataset}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 composite.py <dataset_path> <bool: delete_originals = false> # Needs `camera_1` in camera path, needs `overlay_1` in overlay path")
        sys.exit(1)

    root = Path(sys.argv[1])
    if len(sys.argv) > 2:
        main(root, delete_originals=sys.argv[2].lower() == "true")
    else:
        main(root)
