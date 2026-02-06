import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.model_selection import train_test_split
from PIL import Image
from src.utils import new_resize_img
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from torchvision.ops import box_iou  
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Define paths
csv_path = 'anotaciones_new.csv'
dataset_path = r'your_path'
masks_path = r'your_path'

# Load the CSV
df = pd.read_csv(csv_path)

# Mapping views to labels (class numbers)
view_to_label = {'psax_aov': 1, 'plax': 2, '3c': 3}

# Custom Dataset with verification of box coordinates
class EchoDataset(Dataset):
    def __init__(self, df, dataset_path, masks_path, transform=None):
        self.df = df
        self.dataset_path = dataset_path
        self.masks_path = masks_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        echo_id = row['echo_id']
        frame = row['frame']
        p1 = eval(row['p1'])
        p2 = eval(row['p2'])
        view = row['view']  # The view of the echocardiogram (psax, plax, 3c)
        label = view_to_label[view]  # Get the numeric label of the view

        # Load the frame and the mask
        frame_path = os.path.join(self.dataset_path, echo_id, f'frame_{frame:04d}.npy')
        mask_path = os.path.join(self.masks_path, f'frames_{echo_id}.npy')

        frame = np.load(frame_path)/255.
        mask = np.load(mask_path)
        _, res_mask, _, _, _, _, _ = new_resize_img(frame, mask, 256, 256)

        # Apply the mask
        frame = frame * res_mask

        # Resize the image to 256x256 if it is not already
        frame = Image.fromarray(frame).resize((256, 256))

        if self.transform:
            frame = self.transform(frame)

        # Verify and correct the box coordinates
        x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
        x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])

        # Create the annotations
        boxes = [[x_min, y_min, x_max, y_max]]  # Single object, coordinates (x1, y1, x2, y2)
        labels = [label]  # Assign the label of the view

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)

        return frame, target


# Transformations for the image
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create training and validation Datasets
train_dataset = EchoDataset(train_df, dataset_path, masks_path, transform=transform)
val_dataset = EchoDataset(val_df, dataset_path, masks_path, transform=transform)

# DataLoaders
batch_size = 4
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Pretrained model (Faster-RCNN)
model = models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')

# Modify the model's classifier for the number of classes (1 object plus background)
num_classes = 4  # 3 classes (psax_aov, plax, 3c) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.rpn.nms_thresh = 0.7  # Keep the default value (0.7) or adjust to 0.5 if needed
model.rpn.post_nms_top_n_test = 1  # Only keep one proposed box during inference
model.rpn.post_nms_top_n_train = 1  # Only keep one proposed box during training
model.roi_heads.detections_per_img = 1  # Limit to one prediction per image

# Configuration of optimizer and scheduler (Cosine Annealing with Warm Restarts)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Function to calculate IoU (Intersection over Union)
def calculate_iou(pred_boxes, true_boxes):
    return box_iou(pred_boxes, true_boxes)

# Function to calculate match precision based on IoU
def calculate_match_precision(pred_boxes, true_boxes, iou_threshold=0.5):
    iou = calculate_iou(pred_boxes, true_boxes)
    matches = iou > iou_threshold  # True positives based on IoU
    match_true_positives = matches.sum().item()  # Count true positives
    total_pred = len(pred_boxes)
    
    match_precision = match_true_positives / total_pred if total_pred > 0 else 0
    return match_precision

# Function to calculate label precision
def calculate_label_precision(pred_labels, true_labels):
    correct_labels = (pred_labels == true_labels).sum().item()
    total_pred = len(pred_labels)
    
    label_precision = correct_labels / total_pred if total_pred > 0 else 0
    return label_precision

# Validation function with mAP calculation
def validate_model_with_map(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    match_precisions = []
    label_precisions = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation with separate precision"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Make predictions
            outputs = model(images)
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu()
                pred_labels = output['labels'].cpu()
                true_boxes = targets[i]['boxes'].cpu()
                true_labels = targets[i]['labels'].cpu()

                # Calculate match precision (IoU > threshold)
                match_precision = calculate_match_precision(pred_boxes, true_boxes, iou_threshold)
                match_precisions.append(match_precision)

                # Calculate label precision
                label_precision = calculate_label_precision(pred_labels, true_labels)
                label_precisions.append(label_precision)

    # Calculate average precision for matches and labels
    avg_match_precision = sum(match_precisions) / len(match_precisions) if len(match_precisions) > 0 else 0
    avg_label_precision = sum(label_precisions) / len(label_precisions) if len(label_precisions) > 0 else 0

    # Show results
    print(f"Match Precision (IoU > {iou_threshold}): {avg_match_precision:.4f}")
    print(f"Label Precision: {avg_label_precision:.4f}")
    
    return avg_match_precision, avg_label_precision

# Function to save the model
def save_model(model, epoch, prefix):
    model_path = f"{prefix}/model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

# Function to save results to a text file
def save_training_info(results_dir, epoch, train_loss, avg_match_precision, avg_label_precision):
    info_file = os.path.join(results_dir, f'{results_dir.split(os.sep)[-1]}_info.txt')
    with open(info_file, 'a') as f:
        f.write(f"Epoch {epoch + 1}:\n")
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Match Precision: {avg_match_precision:.4f}\n")
        f.write(f"Label Precision: {avg_label_precision:.4f}\n")
        f.write("="*50 + "\n")

# Function to create results directory
def create_results_directory(base_path='results'):
    current_time = time.strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_path, current_time)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Training function with visualization and model saving
def train_model_with_visualization(model, train_loader, val_loader, optimizer, scheduler=None, num_epochs=30):
    best_val_loss = 0.0  # To track the best model

    # Create results directory
    results_dir = create_results_directory()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # tqdm to show training progress
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass and loss calculation
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Update learning rate using Cosine Annealing scheduler
        if scheduler != None:
            scheduler.step()

        # Total number of training images
        total_images_train = len(train_loader.dataset)

        # Calculate average loss per training image
        avg_train_loss = train_loss / total_images_train

        # Validate the model and calculate mAP on validation set
        avg_match_precision, avg_label_precision = validate_model_with_map(model, val_loader, device)

        # Show progress
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss (per image): {avg_train_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Match Precision: {avg_match_precision:.4f}, Label Precision: {avg_label_precision:.4f}')

        # Save the model after each epoch
        save_model(model, epoch, results_dir)

        # Save training information to text file
        save_training_info(results_dir, epoch, avg_train_loss, avg_match_precision, avg_label_precision)

        # Save the best model (based on match or label precision, whichever you prefer)
        if avg_match_precision > best_val_loss:  # Or use avg_label_precision if you prefer
            best_val_loss = avg_match_precision
            save_model(model, "best", results_dir)

# Train the model with visualization of results
train_model_with_visualization(model, train_data_loader, val_data_loader, optimizer, scheduler=None, num_epochs=30)
