import os
import sys

def download_modelscope():
    print("Attempting to download model from ModelScope...")
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(
            "qwen/Qwen3-VL-Embedding-2B",
            local_dir="./models/Qwen3-VL-Embedding-2B"
        )
        print(f"Model successfully downloaded from ModelScope to {model_dir}")
        return True
    except Exception as e:
        print(f"ModelScope download failed: {e}")
        return False

def download_huggingface():
    print("Attempting to download model from Hugging Face...")
    try:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(
            repo_id="Qwen/Qwen3-VL-Embedding-2B",
            local_dir="./models/Qwen3-VL-Embedding-2B"
        )
        print(f"Model successfully downloaded from Hugging Face to {model_dir}")
        return True
    except Exception as e:
        print(f"Hugging Face download failed: {e}")
        return False

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    if not download_modelscope():
        if not download_huggingface():
            print("Error: Failed to download model from both ModelScope and Hugging Face.")
            sys.exit(1)
