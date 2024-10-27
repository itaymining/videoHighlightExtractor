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
    
    SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
    
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
    
    def list_albums(self) -> List[Dict[str, str]]:
        """
        Retrieve and return list of albums.
        
        Returns:
            List[Dict[str, str]]: List of albums with their IDs and titles
        """
        try:
            albums = self.service.albums().list().execute()
            self._albums_cache = albums.get('albums', [])  # Cache albums for validation
            return self._albums_cache
        except Exception as e:
            logger.error(f"Failed to retrieve albums: {str(e)}")
            raise

    def display_albums(self) -> None:
        """
        Display available albums in a numbered list format.
        """
        print("\nAvailable Albums:")
        print("-" * 50)
        for idx, album in enumerate(self._albums_cache, 1):
            print(f"{idx}. {album['title']}")
        print("-" * 50)

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
                    print(f"\nSelected album: {selected_album['title']}")
                    return selected_album
                else:
                    print("\nError: Invalid selection. Please choose a number from the list.")
            except ValueError:
                print("\nError: Please enter a valid number.")

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
                print("Invalid date format. Please use YYYY-MM-DD format (e.g., 2024-01-31)")
    
    def download_videos(self, album_id: str, start_date: str, end_date: str, 
                       output_dir: str = '.') -> None:
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
                            self._download_video(item, creation_date, output_dir)
                            videos_downloaded += 1
                
                # Check for more pages
                next_page_token = media_items.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing media items: {str(e)}")
                raise
        
        logger.info(f"Total videos downloaded: {videos_downloaded}")
    
    def _download_video(self, media_item: Dict[str, Any], 
                       creation_date: datetime, output_dir: str) -> None:
        """
        Download a single video and save it to the appropriate folder.
        
        Args:
            media_item (Dict[str, Any]): Media item data from Google Photos API
            creation_date (datetime): Creation date of the video
            output_dir (str): Base directory for downloaded videos
        """
        try:
            # Create month-based folder
            month_folder = os.path.join(output_dir, creation_date.strftime('%Y-%m'))
            os.makedirs(month_folder, exist_ok=True)
            
            # Download video
            media_url = f"{media_item['baseUrl']}=dv"
            response = requests.get(media_url)
            response.raise_for_status()
            
            # Save video
            output_path = os.path.join(month_folder, media_item['filename'])
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
        
        # List all albums and get user selection
        downloader.list_albums()
        selected_album = downloader.get_user_selected_album()
        
        # Get date range from user
        print("\nPlease enter the date range for videos to download.")
        print("Use format YYYY-MM-DD (e.g., 2024-01-31)")
        start_date = downloader.get_date_input("Start date: ")
        end_date = downloader.get_date_input("End date: ")
        
        # Download videos
        print("\nStarting download...")
        downloader.download_videos(selected_album['id'], start_date, end_date)
        logger.info("Video download completed successfully")
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
