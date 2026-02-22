#!/usr/bin/env python3
"""
Upload datasets and VAE to Google Drive easily.

Usage:
    python upload_to_drive.py
"""

import os
import sys
from pathlib import Path
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def authenticate():
    """Authenticate with Google Drive"""
    print("🔐 Authenticating with Google Drive...")
    gauth = GoogleAuth()

    # Try to load saved credentials
    if os.path.exists("mycreds.txt"):
        gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.auth_required():
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)
    print("✅ Authenticated!\n")
    return drive

def create_or_get_folder(drive, folder_name, parent_id=None):
    """Create a folder or get it if it exists"""
    query = f"title='{folder_name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    file_list = drive.ListFile({'q': query}).GetList()

    if file_list:
        folder = file_list[0]
        print(f"   📂 Found existing folder: {folder_name}")
        return folder['id']
    else:
        # Create new folder
        folder = drive.CreateFile({
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        })
        if parent_id:
            folder['parents'] = [{'id': parent_id}]
        folder.Upload()
        print(f"   📂 Created folder: {folder_name}")
        return folder['id']

def upload_file(drive, local_path, remote_folder_id):
    """Upload a single file"""
    file_name = os.path.basename(local_path)
    print(f"   📤 Uploading {file_name}...", end=" ", flush=True)

    gfile = drive.CreateFile({
        'title': file_name,
        'parents': [{'id': remote_folder_id}]
    })
    gfile.SetContentFile(local_path)
    gfile.Upload()
    print("✅")

def upload_folder(drive, local_path, remote_folder_id):
    """Upload a folder recursively"""
    local_path = Path(local_path)

    # Create folder on Drive
    folder_name = local_path.name
    folder_id = create_or_get_folder(drive, folder_name, remote_folder_id)

    # Upload all files
    for item in sorted(local_path.rglob('*')):
        if item.is_file():
            relative = item.relative_to(local_path)
            # Create folder structure
            current_folder_id = folder_id
            for parent in relative.parents[:-1]:
                current_folder_id = create_or_get_folder(drive, parent.name, current_folder_id)

            upload_file(drive, str(item), current_folder_id)

def main():
    drive = authenticate()

    # Create root "ai-toolkit" folder
    print("📁 Setting up folders...\n")
    root_folder_id = create_or_get_folder(drive, "ai-toolkit")

    # Upload VAE
    print("\n🎨 Uploading VAE...")
    vae_path = "/home/user/Documents/AlphaVAE/output/local_vae_training/checkpoint-7100"
    if os.path.exists(vae_path):
        upload_folder(drive, vae_path, root_folder_id)
        print("✅ VAE uploaded!\n")
    else:
        print(f"❌ VAE not found at {vae_path}\n")

    # Upload datasets
    print("📊 Uploading datasets...")
    datasets = [
        "/home/user/Documents/ai-toolkit/datasets/composite",
        # Add your other dataset paths here:
        # "/path/to/dataset2",
        # "/path/to/dataset3",
    ]

    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            print(f"\n  {dataset_path}")
            upload_folder(drive, dataset_path, root_folder_id)
        else:
            print(f"\n  ❌ Not found: {dataset_path}")

    print("\n" + "="*60)
    print("✅ Upload complete!")
    print("="*60)
    print(f"\n📂 All files uploaded to Google Drive in 'ai-toolkit' folder")
    print(f"\nUse in Colab:")
    print(f'  VAE_PATH = "/content/drive/My Drive/ai-toolkit/checkpoint-7100"')
    print(f'  DATASET_PATHS = [')
    print(f'      "/content/drive/My Drive/ai-toolkit/composite",')
    print(f'  ]')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
