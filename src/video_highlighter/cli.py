#!/usr/bin/env python3
"""Command-line interface for the AI video highlight generator."""
import argparse
import logging
import sys

from .highlighter import VideoHighlightGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-highlighter",
        description="Analyze a library of videos, extract highlights and build a combined reel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input-dir", dest="input_dirs", action="append", required=True,
        metavar="DIR",
        help="Directory containing input videos (repeat for multiple directories).",
    )
    parser.add_argument(
        "-o", "--output-dir", default="video_highlights",
        help="Directory for output videos.",
    )
    parser.add_argument(
        "--target-duration", type=int, default=300,
        help="Target duration of the combined highlight reel, in seconds.",
    )
    parser.add_argument(
        "--min-clip-duration", type=int, default=2,
        help="Minimum duration of each clip, in seconds.",
    )
    parser.add_argument(
        "--max-clip-duration", type=int, default=3,
        help="Maximum duration of each clip, in seconds.",
    )
    parser.add_argument(
        "--highlights-per-video", type=int, default=2,
        help="Number of top highlights to save per video.",
    )
    parser.add_argument(
        "--max-video-duration", type=int, default=120, dest="max_video_duration_to_process",
        help="Skip videos longer than this many seconds.",
    )
    parser.add_argument(
        "--processing-order", choices=["asc", "desc"], default="asc",
        help="Order in which videos are processed.",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle selected clips instead of keeping processing order.",
    )
    parser.add_argument(
        "--save-metadata", action="store_true",
        help="Write a .txt metadata file next to each highlight.",
    )
    parser.add_argument(
        "--from-existing-clips", action="store_true",
        help="Build the reel from clips already in the output directory.",
    )

    # Face recognition
    face = parser.add_argument_group("face recognition")
    face.add_argument(
        "--use-face-recognition", action="store_true",
        help="Boost segments containing known faces.",
    )
    face.add_argument(
        "--face-images-dir", default="faces", metavar="DIR",
        help="Directory of reference face images (*.jpg).",
    )
    face.add_argument(
        "--has-face-only", action="store_true",
        help="Keep only segments that contain a known face.",
    )
    face.add_argument(
        "--face-detect-min-duration", type=int, default=1,
        help="Minimum seconds a face must appear to count.",
    )
    face.add_argument(
        "--pickle-path", default="known_faces",
        help="Path to cache known-face encodings.",
    )

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        generator = VideoHighlightGenerator(
            input_dirs=args.input_dirs,
            output_dir=args.output_dir,
            target_duration=args.target_duration,
            min_clip_duration=args.min_clip_duration,
            max_clip_duration=args.max_clip_duration,
            face_detect_min_duration=args.face_detect_min_duration,
            save_video_highlight_txt_metadata=args.save_metadata,
            has_face_only=args.has_face_only,
            highlights_per_video=args.highlights_per_video,
            face_images_dir=args.face_images_dir,
            use_face_recongniion=args.use_face_recognition,
            keep_sort_processing_order=not args.shuffle,
            processing_order=args.processing_order,
            pickle_path=args.pickle_path,
            max_video_duration_to_process=args.max_video_duration_to_process,
        )

        if args.from_existing_clips:
            generator.create_highlight_reel_by_existing_clips()
        else:
            generator.create_highlight_reel()

    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
