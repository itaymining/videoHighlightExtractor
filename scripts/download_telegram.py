#!/usr/bin/env python3
"""
Telegram Channel Video Downloader

Downloads videos from a Telegram channel within a date range, skipping duplicates.

Credentials:
    Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env (get them from https://my.telegram.org),
    or pass --api-id / --api-hash on the command line.

Usage:
    python scripts/download_telegram.py --channel @somechannel \
        --start-date 2023-10-07 --end-date 2024-11-01 \
        --max-duration 30 --output-dir ./downloads
"""
import os
import re
import argparse
import asyncio
from datetime import datetime

import pytz
import humanize
from dotenv import load_dotenv
from tqdm import tqdm
from telethon import TelegramClient
from telethon.tl.types import InputMessagesFilterVideo

load_dotenv()


class ProgressCallback:
    def __init__(self, filename, file_size):
        self.pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {filename}")
        self.last_downloaded = 0

    def callback(self, current, total):
        downloaded = current - self.last_downloaded
        self.pbar.update(downloaded)
        self.last_downloaded = current

    def close(self):
        self.pbar.close()


def is_duplicate_file(filepath, output_dir):
    filename = os.path.basename(filepath)
    match = re.match(r'(\d{8})_(\d+)_', filename)
    if not match:
        return False
    date, msg_id = match.groups()
    if os.path.exists(filepath):
        return True
    for existing_file in os.listdir(output_dir):
        if re.match(f"{date}_{msg_id}_", existing_file):
            return True
    return False


async def download_channel_videos(api_id, api_hash, channel_username, start_date, end_date,
                                  max_video_duration=30, output_dir='downloads'):
    os.makedirs(output_dir, exist_ok=True)
    client = TelegramClient('session_name', api_id, api_hash)
    await client.start()

    try:
        channel = await client.get_entity(channel_username)
        messages = []
        async for msg in client.iter_messages(
                channel,
                limit=None,
                reverse=True,
                filter=InputMessagesFilterVideo
        ):
            if start_date <= msg.date <= end_date and \
               (not hasattr(msg.video, 'duration') or msg.video.duration <= max_video_duration):
                messages.append(msg)

        print(f"Found {len(messages)} videos within the specified date range and duration.")

        total_size = sum(msg.video.size for msg in messages if msg.video)
        print(f"Total download size: {humanize.naturalsize(total_size)}")

        successful_downloads = 0
        for index, msg in enumerate(messages, 1):
            date_str = msg.date.strftime('%Y%m%d')
            filename = f"{date_str}_{msg.id}_{msg.video.id}.mp4"
            filepath = os.path.join(output_dir, filename)

            if is_duplicate_file(filepath, output_dir):
                print(f"Skipping duplicate file: {filename}")
                continue

            try:
                progress = ProgressCallback(filename, msg.video.size)
                await client.download_media(
                    msg.video,
                    filepath,
                    progress_callback=progress.callback
                )
                progress.close()
                successful_downloads += 1
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)  # Clean up partial download

        print(f"Successfully downloaded: {successful_downloads}")

    finally:
        await client.disconnect()


def parse_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD into a UTC timezone-aware datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_telegram",
        description="Download videos from a Telegram channel within a date range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-id", default=os.getenv("TELEGRAM_API_ID"),
        help="Telegram API ID (or set TELEGRAM_API_ID in .env).",
    )
    parser.add_argument(
        "--api-hash", default=os.getenv("TELEGRAM_API_HASH"),
        help="Telegram API hash (or set TELEGRAM_API_HASH in .env).",
    )
    parser.add_argument(
        "--channel", required=True,
        help="Channel username, e.g. @somechannel.",
    )
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--max-duration", type=int, default=30,
        help="Maximum video duration to download, in seconds.",
    )
    parser.add_argument(
        "-o", "--output-dir", default="downloads",
        help="Directory for downloaded videos.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.api_id or not args.api_hash:
        raise SystemExit("Missing Telegram credentials. Set TELEGRAM_API_ID and "
                         "TELEGRAM_API_HASH in .env or pass --api-id / --api-hash.")

    asyncio.run(download_channel_videos(
        args.api_id,
        args.api_hash,
        args.channel,
        parse_date(args.start_date),
        parse_date(args.end_date),
        args.max_duration,
        args.output_dir,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
