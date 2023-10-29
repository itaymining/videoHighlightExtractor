import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime
import requests

# Set up API credentials
creds = None
SCOPES = ['https://www.googleapis.com/auth/photoslibrary.readonly']
CLIENT_SECRET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),'sa1.json') # Replace with your credentials file

# If credentials file exists, authorize the client
if os.path.exists('token.json'):
    creds = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRET_FILE, SCOPES).run_local_server(port=0)

# If there are no (valid) credentials available, prompt the user to log in
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)

    # Save the credentials for the next run
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

# Initialize the Google Photos API
service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)

# Retrieve list of albums
albums = service.albums().list().execute()

# Print album names and IDs
for album in albums['albums']:
    print(f"Album Name: {album['title']}")
    print(f"Album ID: {album['id']}")
    print("="*30)

# Define the album ID and date range
album_id = 'AChDSkQ_awFt3KKjgQ3ELHMn7sw177ijnlcKkYgZfgepb6Bon857PFI-Xw7TZi41bdBm0Mq1129p'
start_date = '2022-10-29T00:00:00Z'  # Replace with your start date
end_date = '2022-10-31T23:59:59Z'    # Replace with your end date

next_page_token = None

while True:
    # Retrieve list of media items in the album
    request_body = {'albumId': album_id, 'pageSize': 100}
    if next_page_token:
        request_body['pageToken'] = next_page_token

    # Retrieve list of media items in the album
    media_items = service.mediaItems().search(body=request_body).execute()

    # Download videos
    for media_item in media_items['mediaItems']:
        if 'video' in media_item['mimeType']:
            creation_date = datetime.strptime(media_item['mediaMetadata']['creationTime'], "%Y-%m-%dT%H:%M:%SZ")
            if start_date <= creation_date.isoformat() <= end_date:
                month_folder = creation_date.strftime('%Y-%m')
                if not os.path.exists(month_folder):
                    os.makedirs(month_folder)

                media_url = media_item['baseUrl'] + '=dv'
                response = requests.get(media_url)

                with open(os.path.join(month_folder, f'{media_item["filename"]}'), 'wb') as f:
                    f.write(response.content)
                    print(f'Downloaded: {media_item["filename"]}')
    
    next_page_token = media_items.get('nextPageToken')
    if not next_page_token:
        break
            
