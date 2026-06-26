import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from models.siamese_net import SiameseNetwork

def load_model(model_path):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0) # Add batch dimension
    return img

def compute_distance(model, device, img1_path, img2_path):
    img1 = preprocess_image(img1_path).to(device)
    img2 = preprocess_image(img2_path).to(device)
    
    with torch.no_grad():
        output1 = model.forward_once(img1)
        output2 = model.forward_once(img2)
        
        distance = F.pairwise_distance(output1, output2).item()
        cosine_sim = F.cosine_similarity(output1, output2).item()
        
    return distance, cosine_sim

def evaluate_scripts(model, device, dataset_dir, num_samples=100):
    """
    Evaluates structural similarity across all available script classes by randomly
    sampling character pairs and computing pairwise cosine similarity and Euclidean distance.

    Returns:
        results (dict): Maps (cls_a, cls_b) -> {"cosine_sim": float, "euclidean_dist": float}
    """
    # Collect all available class directories and sample one image per class
    class_dirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    # Build a map: class_name -> list of image paths
    class_images: dict[str, list[str]] = {}
    for cls in class_dirs:
        cls_path = os.path.join(dataset_dir, cls)
        imgs = [
            os.path.join(cls_path, f)
            for f in os.listdir(cls_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if imgs:
            class_images[cls] = imgs

    # Sample a random pair of classes to compute similarity (avoids O(N^2) exhaustive search)
    sampled_classes = random.sample(list(class_images.keys()), min(num_samples, len(class_images)))
    results = {}

    print(f"Evaluating similarity across {len(sampled_classes)} sampled script classes...")
    for i, cls_a in enumerate(sampled_classes):
        for cls_b in sampled_classes[i + 1:]:
            img_a_path = random.choice(class_images[cls_a])
            img_b_path = random.choice(class_images[cls_b])
            dist, cos_sim = compute_distance(model, device, img_a_path, img_b_path)
            results[(cls_a, cls_b)] = {"cosine_sim": cos_sim, "euclidean_dist": dist}

    return results

if __name__ == "__main__":
    model_path = "saved_models/siamese_net.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        exit(1)
        
    model, device = load_model(model_path)
    print(f"Model loaded successfully on {device}")
    print("Testing functionality: Model is ready for influence computation.")
    print("Note: Provide specific image pairs to evaluate_similarity.py to compute the distance metrics.")
