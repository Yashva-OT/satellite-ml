import os
import torch
import torchvision.transforms as transforms
import deepforest
from deepforest.utilities import download_release
import matplotlib.pyplot as plt
from torchgeo.datasets import Landsat8
import ipdb

# Step 1: Download and Load Pretrained DeepForest Model
def download_deepforest_model(model_dir="./models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "DeepForest.pth")
    
    if not os.path.exists(model_path):
        print("Downloading DeepForest model from ArcGIS...")
        model_path = download_release("DeepForest", "neon", model_dir)
    
    return model_path

# Step 2: Load the Pretrained Model
def load_deepforest_model(model_path):
    model = deepforest.DeepForest()
    model.use_release()
    model.model.load_state_dict(torch.load(model_path))
    model.model.eval()
    return model

# Step 3: Load Satellite Image (TorchGeo)
def load_satellite_image():
    dataset = Landsat8(root="./data", split="train", download=True)
    image, _ = dataset[0]  # Get first image sample
    return image

# Step 4: Preprocess Image for DeepForest
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((400, 400)),  # Resize to model's expected input
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).permute(1, 2, 0).numpy()

# Step 5: Perform Inference
def detect_trees(model, image):
    boxes = model.predict_image(image)
    return boxes

# Step 6: Visualize Results
def visualize_results(image, boxes):
    plt.imshow(image)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-')
    plt.title("Tree Detection")
    plt.show()

# Main Execution
if __name__ == "__main__":
    ipdb.set_trace()
    model_path = download_deepforest_model()
    model = load_deepforest_model(model_path)
    
    image = load_satellite_image()
    processed_image = preprocess_image(image)
    
    tree_boxes = detect_trees(model, processed_image)
    visualize_results(processed_image, tree_boxes)
