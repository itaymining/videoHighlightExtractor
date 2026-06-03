#!/usr/bin/env python3
"""
Google Photos Video Downloader

Authenticates with the Google Photos API and downloads videos from a chosen album
within a given date range, organizing them into monthly folders.

Credentials:
    - Place your OAuth client secret JSON in credentials/ (default: credentials/sa1.json),
      or set GOOGLE_CLIENT_SECRET in .env / pass --client-secret.
    - The OAuth token is cached (default: credentials/token.json), override with --token.

Usage:
    python scripts/download_google_photos.py
    python scripts/download_google_photos.py --client-secret credentials/sa1.json \
        --token credentials/token.json --output-dir ./downloads \
        --start-date 2024-01-01 --end-date 2024-12-31
"""
import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GooglePhotosDownloader:
    """Handle authentication and downloading of videos from Google Photos."""

    SCOPES = [
        'https://www.googleapis.com/auth/photoslibrary.readonly',
        'https://www.googleapis.com/auth/photoslibrary.sharing'
    ]

    def __init__(self, client_secret_file: str, token_file: str = 'credentials/token.json'):
        """
        Args:
            client_secret_file (str): Path to the OAuth client secret JSON file.
            token_file (str): Path to save/load the cached token.
        """
        self.client_secret_file = client_secret_file
        self.token_file = token_file
        self.credentials = None
        self.service = None
        self._albums_cache = None

    def authenticate(self) -> None:
        """Authenticate with Google Photos API using OAuth 2.0."""
        try:
            if os.path.exists(self.token_file):
                self.credentials = Credentials.from_authorized_user_file(
                    self.token_file, self.SCOPES)

            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())
                else:
                    if not os.path.exists(self.client_secret_file):
                        raise FileNotFoundError(f"Client secret file not found: {self.client_secret_file}")

                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.client_secret_file, self.SCOPES)
                    self.credentials = flow.run_local_server(port=0)

                # Save the credentials for future runs
                os.makedirs(os.path.dirname(self.token_file) or '.', exist_ok=True)
                with open(self.token_file, 'w') as token:
                    token.write(self.credentials.to_json())

            self.service = build('photoslibrary', 'v1',
                                 credentials=self.credentials,
                                 static_discovery=False)
            logger.info("Successfully authenticated with Google Photos API")

        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise

    @staticmethod
    def parse_datetime(timestamp: str) -> datetime:
        """Parse a Google Photos timestamp string into a datetime object."""
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",      # Format without milliseconds
            "%Y-%m-%dT%H:%M:%S.%fZ"    # Format with milliseconds
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue

        raise ValueError(f"Time data '{timestamp}' does not match any expected formats")

    def list_all_albums(self) -> List[Dict[str, str]]:
        """Retrieve and return list of both owned and shared albums."""
        all_albums = []

        try:
            # Get owned albums
            owned_albums_response = self.service.albums().list().execute()
            owned_albums = owned_albums_response.get('albums', [])
            for album in owned_albums:
                album['type'] = 'owned'
                all_albums.append(album)

            # Get shared albums
            shared_albums_response = self.service.sharedAlbums().list().execute()
            shared_albums = shared_albums_response.get('sharedAlbums', [])
            for album in shared_albums:
                album['type'] = 'shared'
                if 'title' not in album:
                    album['title'] = 'Untitled Shared Album'
                all_albums.append(album)

            self._albums_cache = all_albums
            return all_albums

        except Exception as e:
            logger.error(f"Failed to retrieve albums: {str(e)}")
            raise

    def display_albums(self) -> None:
        """Display available albums in a numbered list (owned and shared)."""
        logger.info("\nAvailable Albums:")
        logger.info("-" * 70)
        logger.info(f"{'#':<4} {'Type':<8} {'Title':<50}")
        logger.info("-" * 70)

        for idx, album in enumerate(self._albums_cache, 1):
            album_type = album.get('type', 'owned')
            logger.info(f"{idx:<4} {album_type:<8} {album['title']:<50} ({album['mediaItemsCount']} items in album {album['id']})")
        logger.info("-" * 70)

    def get_user_selected_album(self) -> Dict[str, str]:
        """Prompt the user to select an album and return it."""
        while True:
            self.display_albums()
            try:
                selection = input("\nEnter the number of the album you want to download from: ")
                idx = int(selection) - 1

                if 0 <= idx < len(self._albums_cache):
                    selected_album = self._albums_cache[idx]
                    logger.info(f"\nSelected album: {selected_album['title']} ({selected_album['type']} album)")
                    return selected_album
                else:
                    logger.info("\nError: Invalid selection. Please choose a number from the list.")
            except ValueError:
                logger.warning("\nError: Please enter a valid number.")

    @staticmethod
    def to_iso(date_str: str) -> str:
        """Validate a YYYY-MM-DD date string and return it in API ISO format."""
        datetime.strptime(date_str, "%Y-%m-%d")
        return f"{date_str}T00:00:00Z"

    def get_date_input(self, prompt: str) -> str:
        """Get and validate a date from the user interactively."""
        while True:
            date_str = input(prompt)
            try:
                return self.to_iso(date_str)
            except ValueError:
                logger.warning("Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-01-31)")

    def download_videos(self, album_id: str, start_date: str, end_date: str, output_dir: str = '.') -> None:
        """Download videos from the specified album within the date range."""
        next_page_token = None
        videos_downloaded = 0

        # Get album title from cache
        album_title = next((album['title'] for album in self._albums_cache if album['id'] == album_id), None)

        while True:
            try:
                # Prepare request for media items
                request_body = {'albumId': album_id, 'pageSize': 100}
                if next_page_token:
                    request_body['pageToken'] = next_page_token

                logger.info(f"Retrieving Media next_page_token: {next_page_token}")

                # Retrieve media items
                media_items = self.service.mediaItems().search(body=request_body).execute()

                # Retrieve initial count of media items
                total_videos = 0
                for item in media_items.get('mediaItems', []):
                    if 'video' in item['mimeType']:
                        creation_time = item['mediaMetadata']['creationTime']
                        creation_date = self.parse_datetime(creation_time)
                        if start_date <= creation_date.isoformat() <= end_date:
                            total_videos += 1

                # Initialize tqdm progress bar
                with tqdm(total=total_videos, desc="Downloading videos", unit="video") as pbar:

                    # Process each media item
                    for item in media_items.get('mediaItems', []):
                        if 'video' in item['mimeType']:
                            creation_time = item['mediaMetadata']['creationTime']
                            creation_date = self.parse_datetime(creation_time)

                            # Check if video is within date range
                            if start_date <= creation_date.isoformat() <= end_date:
                                self._download_video(item, creation_date, output_dir, album_title)
                                videos_downloaded += 1
                                pbar.update(1)

                    # Check for more pages
                    next_page_token = media_items.get('nextPageToken')
                    if not next_page_token:
                        break

            except Exception as e:
                logger.error(f"Error processing media items: {str(e)}")
                raise

        logger.info(f"Total videos downloaded: {videos_downloaded}")

    def _download_video(self, media_item: Dict[str, Any],
                        creation_date: datetime, output_dir: str, album_title: str = None) -> None:
        """Download a single video and save it to the appropriate folder."""
        try:
            # Sanitize album title for file system
            if album_title:
                safe_album_title = "".join(c for c in album_title if c.isalnum() or c in (' ', '-', '_'))
                safe_album_title = safe_album_title.strip()
            else:
                safe_album_title = "Untitled_Album"
            # Create folder with format: YYYY-MM_AlbumName
            folder_name = os.path.join(output_dir, f"{safe_album_title}_{creation_date.strftime('%Y-%m')}")
            os.makedirs(folder_name, exist_ok=True)

            # Download video
            media_url = f"{media_item['baseUrl']}=dv"
            response = requests.get(media_url)
            response.raise_for_status()

            # Save video
            output_path = os.path.join(folder_name, media_item['filename'])
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded: {media_item['filename']}")

        except Exception as e:
            logger.error(f"Failed to download {media_item['filename']}: {str(e)}")
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_google_photos",
        description="Download videos from a Google Photos album within a date range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--client-secret",
        default=os.getenv("GOOGLE_CLIENT_SECRET", "credentials/sa1.json"),
        help="Path to the OAuth client secret JSON (or set GOOGLE_CLIENT_SECRET in .env).",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GOOGLE_TOKEN_FILE", "credentials/token.json"),
        help="Path to the cached OAuth token (created on first run).",
    )
    parser.add_argument(
        "-o", "--output-dir", default=".",
        help="Base directory for downloaded videos.",
    )
    parser.add_argument(
        "--start-date", help="Start date YYYY-MM-DD (prompted if omitted).",
    )
    parser.add_argument(
        "--end-date", help="End date YYYY-MM-DD (prompted if omitted).",
    )
    parser.add_argument(
        "--album-id", help="Album ID to download from (prompted with a list if omitted).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        downloader = GooglePhotosDownloader(args.client_secret, args.token)
        downloader.authenticate()
        downloader.list_all_albums()

        # Resolve album
        if args.album_id:
            album_id = args.album_id
        else:
            album_id = downloader.get_user_selected_album().get('id')

        # Resolve date range
        if args.start_date and args.end_date:
            start_date = downloader.to_iso(args.start_date)
            end_date = downloader.to_iso(args.end_date)
        else:
            logger.info("\nPlease enter the date range for videos to download.")
            logger.info("Use format YYYY-MM-DD (e.g., 2024-01-31)")
            start_date = downloader.get_date_input("Start date: ")
            end_date = downloader.get_date_input("End date: ")

        logger.info("\nStarting download...")
        downloader.download_videos(album_id, start_date, end_date, args.output_dir)
        logger.info("Video download completed successfully")

    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
