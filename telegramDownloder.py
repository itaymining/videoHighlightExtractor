from telethon import TelegramClient, sync, events
from datetime import datetime
import os
import asyncio
from tqdm import tqdm
import humanize
import re
from telethon.tl.types import MessageMediaDocument, InputMessagesFilterVideo
import pytz

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

def main():
    API_ID = "id"
    API_HASH = "hash"
    
    CHANNEL_USERNAME = "@amitsegal"
    # Define timezone-aware start and end dates
    START_DATE = datetime(2023, 10, 7, tzinfo=pytz.UTC)  # Year, Month, Day
    END_DATE = datetime(2024, 11, 1, tzinfo=pytz.UTC)    # Year, Month, Day
    MAX_VIDEO_DURATION = 30
    
    asyncio.run(download_channel_videos(
        API_ID,
        API_HASH,
        CHANNEL_USERNAME,
        START_DATE,
        END_DATE,
        MAX_VIDEO_DURATION
    ))

if __name__ == "__main__":
    main()
