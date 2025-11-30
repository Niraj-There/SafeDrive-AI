"""
Model Download Helper Script

This script automatically downloads required models for SafeDrive-AI.
Run this after cloning the repository.
"""

import os
import sys
import urllib.request
from tqdm import tqdm

# Model download URLs
MODEL_URLS = {
    'accident_model.pth': 'https://www.kaggle.com/api/v1/models/nirajthere/accident-detection-and-classification-model/other/default/1/download',
    'yolov8m.pt': 'https://www.kaggle.com/api/v1/models/nirajthere/yolov8-detects-and-tracks-objects-in-dashcam-videos/other/default/1/download'
}

def download_file(url, filename):
    """Download file with progress bar"""
    try:
        print(f"📥 Downloading {filename}...")
        
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename, reporthook=t.update_to)
        
        print(f"✅ {filename} downloaded successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        print(f"   Please download manually from: {url}\n")
        return False

print("=" * 70)
print("SafeDrive-AI - Automatic Model Downloader")
print("=" * 70)
print()

# Check which models are missing (only required models)
models_status = {
    'accident_model.pth': os.path.exists('accident_model.pth'),
    'yolov8m.pt': os.path.exists('yolov8m.pt')
}

print("Checking model files...")
print()

all_present = True
for model, exists in models_status.items():
    if exists:
        size_mb = os.path.getsize(model) / (1024 * 1024)
        print(f"✅ {model} - Found ({size_mb:.1f} MB)")
    else:
        print(f"❌ {model} - MISSING")
        all_present = False

print()

if all_present:
    print("🎉 All models are present! You're ready to go!")
    print()
    print("Run this to test:")
    print("  python test_project.py")
    sys.exit(0)

# Ask user if they want to download
print("=" * 70)
print("MISSING MODELS DETECTED")
print("=" * 70)
print()

missing_models = [model for model, exists in models_status.items() if not exists]

if missing_models:
    print(f"Found {len(missing_models)} missing model(s):")
    for model in missing_models:
        print(f"  • {model}")
    print()
    
    response = input("Do you want to download them? (y/n): ").lower().strip()
    print()
    
    if response == 'y' or response == 'yes':
        print("=" * 70)
        print("DOWNLOADING MODELS...")
        print("=" * 70)
        print()
        
        success_count = 0
        for model in missing_models:
            if model in MODEL_URLS:
                if download_file(MODEL_URLS[model], model):
                    success_count += 1
        
        print("=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
        print(f"Successfully downloaded: {success_count}/{len(missing_models)} models")
        print()
        
        if success_count == len(missing_models):
            print("🎉 All models downloaded successfully!")
            print()
            print("Next step:")
            print("  python test_project.py")
        else:
            print("⚠️  Some downloads failed. Please download manually:")
            for model in missing_models:
                if not os.path.exists(model) and model in MODEL_URLS:
                    print(f"  • {model}: {MODEL_URLS[model]}")
    else:
        print("=" * 70)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("=" * 70)
        print()
        
        if not models_status['accident_model.pth']:
            print("1. accident_model.pth (CNN Accident Classifier)")
            print("   📥 Kaggle: https://www.kaggle.com/api/v1/models/nirajthere/accident-detection-and-classification-model/other/default/1/download")
            print()
        
        if not models_status['yolov8m.pt']:
            print("2. yolov8m.pt (YOLO Object Detector)")
            print("   📥 Kaggle: https://www.kaggle.com/api/v1/models/nirajthere/yolov8-detects-and-tracks-objects-in-dashcam-videos/other/default/1/download")
            print()
        
        print("Windows PowerShell:")
        print('  Invoke-WebRequest -Uri "URL" -OutFile "filename.pth"')
        print()
        print("Linux/Mac:")
        print('  wget URL -O filename.pth')
        print()

print()
print("=" * 70)
print("After downloading, run this script again to verify.")
print("=" * 70)
