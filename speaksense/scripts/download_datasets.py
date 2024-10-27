import os
import argparse
import subprocess
import requests
import kaggle
import gdown
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm

class DatasetDownloader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.datasets_path = self.base_path / "datasets"
        self.datasets_path.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filename):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def download_voxceleb(self):
        """Download VoxCeleb dataset"""
        print("Downloading VoxCeleb dataset...")
        
        # You need to register at http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
        vox_path = self.datasets_path / "voxceleb"
        vox_path.mkdir(exist_ok=True)
        
        print("Please download VoxCeleb from: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/")
        print("After obtaining credentials, place the files in:", vox_path)

    def download_slurp(self):
        """Download SLURP dataset"""
        print("Downloading SLURP dataset...")
        slurp_path = self.datasets_path / "slurp"
        slurp_path.mkdir(exist_ok=True)
        
        # Download using gdown (Google Drive)
        slurp_url = "https://drive.google.com/uc?id=1Qj7VhJ7YYMASTp_HHeDOjXK3h5LDvxeY"
        output = slurp_path / "slurp.tar.gz"
        
        gdown.download(slurp_url, str(output), quiet=False)
        
        # Extract
        with tarfile.open(output) as tar:
            tar.extractall(path=slurp_path)
        
        # Clean up
        output.unlink()

    def download_columbia_gaze(self):
        """Download Columbia Gaze dataset"""
        print("Downloading Columbia Gaze dataset...")
        gaze_path = self.datasets_path / "columbia_gaze"
        gaze_path.mkdir(exist_ok=True)
        
        # Download from Columbia's server
        url = "http://www.cs.columbia.edu/CAVE/databases/columbia_gaze/columbia_gaze_data.zip"
        output = gaze_path / "columbia_gaze.zip"
        
        self.download_file(url, output)
        
        # Extract
        with zipfile.ZipFile(output) as zip_ref:
            zip_ref.extractall(gaze_path)
            
        # Clean up
        output.unlink()

    def download_mit_open_voice(self):
        """Download MIT Open Voice dataset"""
        print("Downloading MIT Open Voice dataset...")
        voice_path = self.datasets_path / "mit_open_voice"
        voice_path.mkdir(exist_ok=True)
        
        print("Please visit: https://openvoice.mit.edu/ to request access to the dataset")
        print("After obtaining access, place the files in:", voice_path)

def main():
    parser = argparse.ArgumentParser(description='Download datasets for SpeakSense')
    parser.add_argument('--data-path', type=str, default='data',
                      help='Base path for downloading datasets')
    parser.add_argument('--datasets', nargs='+', 
                      choices=['voxceleb', 'slurp', 'columbia_gaze', 'mit_open_voice'],
                      default=['all'],
                      help='Specific datasets to download')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_path)
    
    if 'all' in args.datasets:
        datasets = ['voxceleb', 'slurp', 'columbia_gaze', 'mit_open_voice']
    else:
        datasets = args.datasets
    
    for dataset in datasets:
        if dataset == 'voxceleb':
            downloader.download_voxceleb()
        elif dataset == 'slurp':
            downloader.download_slurp()
        elif dataset == 'columbia_gaze':
            downloader.download_columbia_gaze()
        elif dataset == 'mit_open_voice':
            downloader.download_mit_open_voice()

if __name__ == "__main__":
    main()