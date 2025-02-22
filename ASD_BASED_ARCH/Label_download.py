import os
import subprocess
import gdown
import tarfile

def download_labels(video_folder):
    label_folder = os.path.join(video_folder, "col_labels")
    label_tar_path = os.path.join(video_folder, "col_labels.tar.gz")
    
    # Google Drive file ID (Make sure it's correct)
    gdrive_id = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"

    # Check if labels are already downloaded
    if not os.path.isdir(label_folder):
        print("Downloading labels...")

        try:
            # Download using gdown
            gdown.download(id=gdrive_id, output=label_tar_path, quiet=False)

            # Check if file exists
            if not os.path.exists(label_tar_path) or os.path.getsize(label_tar_path) == 0:
                print("Download failed or file is empty.")
                return

            # Extract the downloaded tar.gz file
            print("Extracting labels...")
            with tarfile.open(label_tar_path, "r:gz") as tar:
                tar.extractall(path=video_folder)

            # Remove the tar.gz file after extraction
            os.remove(label_tar_path)
            print("Labels downloaded and extracted successfully.")

        except Exception as e:
            print("Error:", e)

# Example usage
video_folder = r"C:\Users\Rohit Francis\Desktop\Codes\TESTINGS\Diarization Test\Light-ASD\VideoFolder"
download_labels(video_folder)
