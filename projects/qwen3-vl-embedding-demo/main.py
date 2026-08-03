import os
import sys
import torch

# Add the project root to python path so we can import src.models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

def run_demo():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "Qwen3-VL-Embedding-2B")
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}.")
        print("Please run 'python download_model.py' first.")
        return

    print("Initializing Qwen3-VL-Embedding-2B...")
    # Detect device
    device_type = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_type}")

    # Use float16 for MPS/CUDA to save memory and run faster, CPU uses float32
    torch_dtype = torch.float16 if device_type in ["mps", "cuda"] else torch.float32

    model = Qwen3VLEmbedder(
        model_name_or_path=model_path,
        torch_dtype=torch_dtype
    )

    print("Generating embeddings for sample inputs...")
    inputs = [
        # Query 1
        {
            "text": "A woman playing with her dog on a beach at sunset.",
            "instruction": "Retrieve images or text relevant to the user's query.",
        },
        # Document 1 (relevant text)
        {
            "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
        },
        # Document 2 (relevant image)
        {
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        },
        # Document 3 (unrelated text)
        {
            "text": "A close-up shot of a modern office space with multiple computers and screens."
        }
    ]

    embeddings = model.process(inputs)
    print("\nEmbeddings generated successfully!")
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute similarity matrix
    similarity_matrix = embeddings @ embeddings.T
    print("\nCosine Similarity Matrix:")
    print(similarity_matrix.cpu().numpy())

    print("\nResults Analysis (Relative to Query 1):")
    # First item is the query
    query_name = "Query: A woman playing with her dog on a beach at sunset"
    doc_names = [
        "Doc 1 (Text): A woman shares a joyful moment with her golden retriever...",
        "Doc 2 (Image): Sunset beach dog image...",
        "Doc 3 (Text): A close-up shot of a modern office space..."
    ]

    for i, doc_name in enumerate(doc_names):
        sim = similarity_matrix[0, i + 1].item()
        print(f" - Similarity with {doc_name}: {sim:.4f}")

if __name__ == "__main__":
    run_demo()
