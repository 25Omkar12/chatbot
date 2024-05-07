import io
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_file(file_id, credentials_file):
    try:
        # Authenticate with the Google Drive API using service account credentials
        credentials = service_account.Credentials.from_service_account_file(credentials_file)
        service = build('drive', 'v3', credentials=credentials)

        # Retrieve file metadata
        file_metadata = service.files().get(fileId=file_id).execute()
        original_file_name = file_metadata['name']
        
        # Get the folder path from the environment variable
        download_folder = os.getenv("DOWNLOAD_FOLDER")

        # Download file content
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f'Downloading {int(status.progress() * 100)}%')

        # Save downloaded content to a file in the specified folder
        file_path = os.path.join(download_folder, original_file_name)
        with open(file_path, 'wb') as f:
            f.write(fh.getvalue())
        
        # Rename the file to "data.pdf"
        new_file_path = os.path.join(download_folder, "data.pdf")
        os.rename(file_path, new_file_path)
        
        print(f'File downloaded successfully: {new_file_path}')
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    # Replace with the file ID and path to the service account credentials JSON file
    file_id = '1OBjCEoYTwanYThbQ-jper7cPII_7jBfY'
    credentials_file = os.getenv("CREDENTIALS_FILE")
    
    download_file(file_id, credentials_file)
