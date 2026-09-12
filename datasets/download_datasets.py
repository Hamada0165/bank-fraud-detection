"""Script to download Porto Seguro and IEEE-CIS Fraud Detection datasets from Kaggle"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

# Resolves the kaggle CLI from PATH (pip install kaggle puts it there).
KAGGLE_EXE = shutil.which("kaggle") or "kaggle"
DATASETS_DIR = Path(__file__).parent

def download_dataset(competition_name, dataset_name):
    """Download a Kaggle competition dataset"""
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}...")
    print(f"{'='*60}")

    cmd = [
        KAGGLE_EXE,
        "competitions", "download",
        "-c", competition_name
    ]

    try:
        # Change to datasets directory
        os.chdir(DATASETS_DIR)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully downloaded {dataset_name}")

            # Find and unzip the downloaded file
            zip_files = list(DATASETS_DIR.glob(f"{competition_name}.zip"))
            if zip_files:
                zip_path = zip_files[0]
                print(f"Extracting {zip_path.name}...")

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract to a folder with dataset name
                    extract_dir = DATASETS_DIR / dataset_name
                    extract_dir.mkdir(exist_ok=True)
                    zip_ref.extractall(extract_dir)

                print(f"Extracted to {extract_dir}")
                # Optionally remove zip file to save space
                # zip_path.unlink()

            return True
        else:
            print(f"Error downloading {dataset_name}:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

def main():
    print("Kaggle Dataset Downloader")
    print("=" * 60)

    # Check if kaggle.json exists
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\nWARNING: Kaggle credentials not found!")
        print(f"Expected location: {kaggle_json}")
        print("\nPlease set up Kaggle API credentials first — see datasets/README.md.")
        print("\nAttempting download anyway...")
    else:
        print(f"Kaggle credentials found at {kaggle_json}")

    # Create datasets directory if it doesn't exist
    DATASETS_DIR.mkdir(exist_ok=True)

    # Download datasets
    datasets = [
        ("porto-seguro-safe-driver-prediction", "porto_seguro"),
        ("ieee-fraud-detection", "ieee_fraud")
    ]

    results = []
    for competition, dataset_name in datasets:
        success = download_dataset(competition, dataset_name)
        results.append((dataset_name, success))

    # Summary
    print("\n" + "="*60)
    print("Download Summary:")
    print("="*60)
    for dataset_name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{dataset_name}: {status}")

    if all(success for _, success in results):
        print("\nAll datasets downloaded successfully!")
    else:
        print("\nSome downloads failed. Check errors above.")

if __name__ == "__main__":
    main()
