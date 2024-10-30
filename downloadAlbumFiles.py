#!/usr/bin/env python3
"""
Google Photos Video Downloader

This script authenticates with Google Photos API and downloads videos from a specified album
within a given date range, organizing them into monthly folders.

Required dependencies:
    - google-auth-oauthlib
    - google-api-python-client
    - requests
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GooglePhotosDownloader:
    """
    A class to handle authentication and downloading of videos from Google Photos.
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/photoslibrary.readonly',
        'https://www.googleapis.com/auth/photoslibrary.sharing'
    ]
    
    def __init__(self, client_secret_file: str, token_file: str = 'token.json'):
        """
        Initialize the downloader with credentials files.
        
        Args:
            client_secret_file (str): Path to the client secret JSON file
            token_file (str): Path to save/load the token
        """
        self.client_secret_file = client_secret_file
        self.token_file = token_file
        self.credentials = None
        self.service = None
        self._albums_cache = None  # Cache for albums to validate user input
    
    def authenticate(self) -> None:
        """
        Authenticate with Google Photos API using OAuth 2.0.
        
        Raises:
            FileNotFoundError: If client secret file doesn't exist
            Exception: For other authentication errors
        """
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
        """
        Parse timestamp string into datetime object.
        
        Args:
            timestamp (str): Timestamp string from Google Photos API
            
        Returns:
            datetime: Parsed datetime object
            
        Raises:
            ValueError: If timestamp format is not recognized
        """
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
        """
        Retrieve and return list of both owned and shared albums.
        
        Returns:
            List[Dict[str, str]]: List of all albums with their IDs and titles
        """
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
                if not 'title' in album:
                    album['title'] = 'Untitled Shared Album'
                all_albums.append(album)
            
            self._albums_cache = all_albums
            return all_albums
            
        except Exception as e:
            logger.error(f"Failed to retrieve albums: {str(e)}")
            raise

    def display_albums(self) -> None:
        """
        Display available albums in a numbered list format, showing both owned and shared albums.
        """
        logger.info("\nAvailable Albums:")
        logger.info("-" * 70)
        logger.info(f"{'#':<4} {'Type':<8} {'Title':<50}")
        logger.info("-" * 70)
        
        for idx, album in enumerate(self._albums_cache, 1):
            album_type = album.get('type', 'owned')
            logger.info(f"{idx:<4} {album_type:<8} {album['title']:<50} ({album['mediaItemsCount']} items in album {album['id']})")
        logger.info("-" * 70)

    def get_user_selected_album(self) -> Dict[str, str]:
        """
        Prompt user to select an album from the list and return the selected album.
        
        Returns:
            Dict[str, str]: Selected album information
            
        Raises:
            ValueError: If invalid selection is made
        """
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

    def get_date_input(self, prompt: str) -> str:
        """
        Get and validate date input from user.
        
        Args:
            prompt (str): Prompt message for the user
            
        Returns:
            str: Validated date string in ISO format
        """
        while True:
            date_str = input(prompt)
            try:
                # Try to parse the date to validate it
                datetime.strptime(date_str, "%Y-%m-%d")
                # Return in proper format for the API
                return f"{date_str}T00:00:00Z"
            except ValueError:
                logger.warning("Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-01-31)")
    
    def download_videos(self, album_id: str, start_date: str, end_date: str, output_dir: str = '.') -> None:
        """
        Download videos from specified album within date range.
        
        Args:
            album_id (str): ID of the Google Photos album
            start_date (str): Start date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
            end_date (str): End date in ISO format (YYYY-MM-DDTHH:MM:SSZ)
            output_dir (str): Base directory for downloaded videos
            
        Raises:
            Exception: If download or file operations fail
        """
        next_page_token = None
        videos_downloaded = 0

        # Get album title from cache
        album_title = next((album['title'] for album in self._albums_cache if album['id'] == album_id), None)
        
        # Retrieve initial count of media items
        total_videos = 0
        media_items_preview = self.service.mediaItems().search(
            body={'albumId': album_id, 'pageSize': 100}
        ).execute()
        for item in media_items_preview.get('mediaItems', []):
            if 'video' in item['mimeType']:
                creation_time = item['mediaMetadata']['creationTime']
                creation_date = self.parse_datetime(creation_time)
                if start_date <= creation_date.isoformat() <= end_date:
                    total_videos += 1

        # Initialize tqdm progress bar
        with tqdm(total=total_videos, desc="Downloading videos", unit="video") as pbar:
            while True:
                try:
                    # Prepare request for media items
                    request_body = {'albumId': album_id, 'pageSize': 100}
                    if next_page_token:
                        request_body['pageToken'] = next_page_token
                    
                    # Retrieve media items
                    media_items = self.service.mediaItems().search(body=request_body).execute()
                    
                    # Process each media item
                    for item in media_items.get('mediaItems', []):
                        if 'video' in item['mimeType']:
                            creation_time = item['mediaMetadata']['creationTime']
                            creation_date = self.parse_datetime(creation_time)
                            
                            # Check if video is within date range
                            if start_date <= creation_date.isoformat() <= end_date:
                                self._download_video(item, creation_date, output_dir, album_title)
                                videos_downloaded += 1
                                pbar.update(1)  # Update the progress bar
                    
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
        """
        Download a single video and save it to the appropriate folder.
        
        Args:
            media_item (Dict[str, Any]): Media item data from Google Photos API
            creation_date (datetime): Creation date of the video
            output_dir (str): Base directory for downloaded videos
        """
        try:
            # Sanitize album title for file system
            if album_title:
                # Remove any characters that might cause issues in file paths
                safe_album_title = "".join(c for c in album_title if c.isalnum() or c in (' ', '-', '_'))
                safe_album_title = safe_album_title.strip()
            else:
                safe_album_title = "Untitled_Album"
            # Create folder with format: YYYY-MM_AlbumName
            folder_name = f"{safe_album_title}_{creation_date.strftime('%Y-%m')}"
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

def main():
    """
    Main function to run the video downloader.
    """
    # Configuration
    CLIENT_SECRET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sa1.json')
    
    try:
        # Initialize and authenticate
        downloader = GooglePhotosDownloader(CLIENT_SECRET_FILE)
        downloader.authenticate()
        
        # List all albums (including shared ones) and get user selection
        downloader.list_all_albums()
        selected_album = downloader.get_user_selected_album()
        
        # Get date range from user
        logger.info("\nPlease enter the date range for videos to download.")
        logger.info("Use format YYYY-MM-DD (e.g., 2024-01-31)")
        start_date = downloader.get_date_input("Start date: ")
        end_date = downloader.get_date_input("End date: ")
        
        # Download videos
        logger.info("\nStarting download...")
        album_id = selected_album.get('id')
    
        downloader.download_videos(album_id, start_date, end_date)
        logger.info("Video download completed successfully")
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
