#!/usr/bin/env python3
"""
Slow-Motion to Normal-Speed Converter

Re-times slow-motion MP4 files to a normal frame rate using FFmpeg.

Usage:
    python scripts/convert_slowmo.py --input-folder ./slowmo --output-folder ./normal
    python scripts/convert_slowmo.py -i ./slowmo -o ./normal --target-fps 30
"""
import os
import time
import argparse

import ffmpeg
from tqdm import tqdm


def normalize_speed(input_file, output_file, target_fps=30):
    """Convert a video to normal speed and save it."""
    print(f"Processing {input_file} -> {output_file}")

    try:
        # Get video metadata
        probe = ffmpeg.probe(input_file)
    except ffmpeg.Error as e:
        print(f"Error with {input_file}: {e}")
        return
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

    if not video_stream:
        print(f"⚠️ No video stream found in {input_file}")
        return

    original_fps = eval(video_stream['r_frame_rate'])  # Convert frame rate to float
    speed_factor = original_fps / target_fps

    if speed_factor == 1:
        print(f"✅ {os.path.basename(input_file)} is already at normal speed.")
        return

    # Get duration for progress bar
    duration = float(video_stream.get('duration', 0))

    # Process with FFmpeg and show progress
    with tqdm(total=duration, unit="s", desc=f"Processing {os.path.basename(input_file)}", dynamic_ncols=True) as pbar:
        process = (
            ffmpeg.input(input_file)
            .filter("setpts", f"{1/speed_factor}*PTS")
            .output(output_file, r=target_fps, vcodec="libx264", preset="fast")
            .run_async(pipe_stderr=True)
        )

        while process.poll() is None:
            time.sleep(1)
            pbar.update(1)

    print(f"✅ Converted: {output_file}")


def batch_process(input_folder, output_folder, target_fps=30):
    """Process all MP4 files in a folder and save them to an output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mp4_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]

    if not mp4_files:
        print("⚠️ No MP4 files found in the input folder.")
        return

    for file in mp4_files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        normalize_speed(input_path, output_path, target_fps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert_slowmo",
        description="Re-time slow-motion MP4 files to a normal frame rate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input-folder", required=True, help="Folder of MP4 files to convert.")
    parser.add_argument("-o", "--output-folder", required=True, help="Folder for converted videos.")
    parser.add_argument("--target-fps", type=int, default=30, help="Target frame rate.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    batch_process(args.input_folder, args.output_folder, args.target_fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
