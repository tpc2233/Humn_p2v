# download_models.py

import os
from huggingface_hub import snapshot_download

def download_models():
    """
    Downloads pre-trained models from Hugging Face Hub to the ./checkpoints directory.
    """
    # --- Configuration ---
    # The base directory where all models will be stored
    checkpoints_base_dir = "SCAIL-Preview"

    # A dictionary mapping the desired local folder name to the Hugging Face repo ID
    models_to_download = {
        "SCAIL-Preview": "zai-org/SCAIL-Preview",
    }
    
    # --- Main Logic ---
    print("--- Starting Model Download Script ---")
    
    # Ensure the base checkpoints directory exists
    print(f"Ensuring base directory '{checkpoints_base_dir}' exists...")
    os.makedirs(checkpoints_base_dir, exist_ok=True)

    # Loop through the dictionary and download each model
    for local_name, repo_id in models_to_download.items():
        print("\n" + "="*50)
        print(f"Processing model: {local_name} ({repo_id})")
        
        # Define the full path for the model
        model_path = checkpoints_base_dir
        
        # Check if the directory already exists and has content to avoid re-downloading
        if os.path.exists(model_path) and os.listdir(model_path):
            print(f"Directory '{model_path}' already exists and is not empty. Skipping download.")
            print("To force a re-download, please delete this directory and run the script again.")
            continue
        
        print(f"Downloading model from {repo_id} to '{model_path}'...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,  # Set to False to copy files directly
                # You can add 'allow_patterns' or 'ignore_patterns' here if you
                # only want to download specific files, e.g., allow_patterns=["*.safetensors"]
            )
            print(f"Successfully downloaded {local_name}.")
        except Exception as e:
            print(f"An error occurred while downloading {repo_id}: {e}")

    print("\n" + "="*50)
    print("--- Model download process finished. ---")
    print(f"All models are located in the './{checkpoints_base_dir}' directory.")


if __name__ == "__main__":
    download_models()
