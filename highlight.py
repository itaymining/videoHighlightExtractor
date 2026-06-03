#!/usr/bin/env python3
"""Runnable entry point for the video highlight generator.

Usage:
    python highlight.py --input-dir ./my_videos --output-dir ./out
    python highlight.py -i ./vids -i ./more_vids --target-duration 300
    python highlight.py --help
"""
import os
import sys

# Make the src/ package importable without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from video_highlighter.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
