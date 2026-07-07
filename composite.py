import subprocess
import sys

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 composite.py <raw_video> <robot_overlay>")
        sys.exit(1)
    raw_video = sys.argv[1]
    overlay_video = sys.argv[2]
    output_path = "composite.mp4"
    subprocess.run([
        "ffmpeg",
        "-i", str(raw_video),       # background (raw video)
        "-i", str(overlay_video),   # foreground (robot on red background)
        "-filter_complex",
        "[1:v]colorkey=0xFF0000:0.3:0.1[fg];[0:v][fg]overlay",
        "-c:v", "libx264",
        str(output_path),
    ], check=True)
