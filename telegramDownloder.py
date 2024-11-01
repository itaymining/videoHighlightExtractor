from telethon import TelegramClient, sync, events
from datetime import datetime
import os
import asyncio
from tqdm import tqdm
import humanize
from telethon.tl.types import MessageMediaDocument

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

async def download_channel_videos(api_id, api_hash, channel_username, start_date, end_date, output_dir='downloads'):
    """
    Download videos from a Telegram channel within a specified date range.
    
    Parameters:
    api_id (int): Telegram API ID from https://my.telegram.org
    api_hash (str): Telegram API hash from https://my.telegram.org
    channel_username (str): Username or link of the public channel
    start_date (datetime): Start date for video download
    end_date (datetime): End date for video download
    output_dir (str): Directory to save downloaded videos
    """
    print("\n=== Telegram Channel Video Downloader ===")
    print(f"Channel: {channel_username}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Output Directory: {output_dir}\n")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the client
    print("Connecting to Telegram...")
    client = TelegramClient('session_name', api_id, api_hash)
    await client.start()
    print("Connected successfully!")
    
    try:
        # Get the channel entity
        print(f"\nFetching channel information...")
        channel = await client.get_entity(channel_username)
        print(f"Channel found: {channel.title}")
        
        # Get all messages from the channel
        print("\nFetching messages...")
        messages = await client.get_messages(channel, limit=None)
        print(f"Total messages found: {len(messages)}")
        
        # Filter messages within date range that contain videos
        print("\nFiltering video messages...")
        video_messages = [
            msg for msg in messages 
            if msg.date >= start_date and msg.date <= end_date 
            and msg.video is not None
        ]
        
        print(f"\nFound {len(video_messages)} videos within the specified date range.")
        
        # Calculate total size
        total_size = sum(msg.video.size for msg in video_messages)
        print(f"Total download size: {humanize.naturalsize(total_size)}")
        
        # Initialize counters
        successful_downloads = 0
        skipped_downloads = 0
        failed_downloads = 0
        
        # Download each video
        print("\nStarting downloads...")
        for index, msg in enumerate(video_messages, 1):
            # Generate filename with date and message ID
            date_str = msg.date.strftime('%Y%m%d')
            filename = f"{date_str}_{msg.id}_{msg.video.id}.mp4"
            filepath = os.path.join(output_dir, filename)
            
            print(f"\nProcessing video {index}/{len(video_messages)}")
            print(f"Date: {msg.date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Size: {humanize.naturalsize(msg.video.size)}")
            
            # Skip if file already exists
            if os.path.exists(filepath):
                print(f"Skipping {filename} (already exists)")
                skipped_downloads += 1
                continue
            
            try:
                progress = ProgressCallback(filename, msg.video.size)
                await client.download_media(
                    msg.video,
                    filepath,
                    progress_callback=progress.callback
                )
                progress.close()
                print(f"Successfully downloaded: {filename}")
                successful_downloads += 1
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                failed_downloads += 1
                if os.path.exists(filepath):
                    os.remove(filepath)  # Clean up partial download
                continue
            
        # Print summary
        print("\n=== Download Summary ===")
        print(f"Total videos found: {len(video_messages)}")
        print(f"Successfully downloaded: {successful_downloads}")
        print(f"Skipped (already existed): {skipped_downloads}")
        print(f"Failed downloads: {failed_downloads}")
        print(f"Total size processed: {humanize.naturalsize(total_size)}")
            
    finally:
        await client.disconnect()
        print("\nDisconnected from Telegram")

def main():
    # Your Telegram API credentials (get from https://my.telegram.org)
    API_ID = "id"
    API_HASH = "hash"
    
    # Channel and date range configuration
    CHANNEL_USERNAME = "@amitsegal"
    START_DATE = datetime(2023, 10, 7)  # Year, Month, Day
    END_DATE = datetime(2024, 11, 1)    # Year, Month, Day
    
    # Run the downloader
    asyncio.run(download_channel_videos(
        API_ID,
        API_HASH,
        CHANNEL_USERNAME,
        START_DATE,
        END_DATE
    ))

if __name__ == "__main__":
    main()