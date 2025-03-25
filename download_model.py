from transformers import CLIPModel, CLIPProcessor
import os

MODEL_DIR = "d:/RAG_Chatbot/models/clip-vit-base-patch16"

def download_and_save_model():
    """Download and save the CLIP model locally"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print("Downloading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        model.save_pretrained(MODEL_DIR)
        processor.save_pretrained(MODEL_DIR)
        print(f"Model saved to {MODEL_DIR}")
    else:
        print("Model already exists at", MODEL_DIR)

if __name__ == "__main__":
    download_and_save_model()