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
from torchvision.ops import box_iou  # Para calcular IoU entre cajas predichas y reales
import time
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Define paths
csv_path = 'data/anotaciones_new_converted.csv'
dataset_path = r'\\NAS3_Z\all\BKP_PERE\echoqual\data\frames_resized\256_256_new'
masks_path = r'e:\25366074H\Documents\echoqual\data\masks'

# Cargar el CSV
df = pd.read_csv(csv_path)

# Mapeo de las vistas a etiquetas (números de clases)
view_to_label = {'psax_aov': 1, 'plax': 2, '3c': 3}

# Custom Dataset con verificación de las coordenadas de las cajas
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
        view = row['view']  # La vista del ecocardiograma (psax, plax, 3c)
        label = view_to_label[view]  # Obtener la etiqueta numérica de la vista

        # Cargar el frame y la máscara
        frame_path = os.path.join(self.dataset_path, echo_id, f'frame_{frame:04d}.npy')
        mask_path = os.path.join(self.masks_path, f'frames_{echo_id}.npy')

        frame = np.load(frame_path)/255.
        mask = np.load(mask_path)
        _, res_mask, _, _, _, _, _ = new_resize_img(frame, mask, 256, 256)

        # Aplicar la máscara
        frame = frame * res_mask

        # Redimensionar la imagen a 256x256 si no lo está
        frame = Image.fromarray(frame).resize((256, 256))

        if self.transform:
            frame = self.transform(frame)

        # Verificar y corregir las coordenadas de la caja
        x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
        x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])

        # Crear las anotaciones
        boxes = [[x_min, y_min, x_max, y_max]]  # Un solo objeto, las coordenadas (x1, y1, x2, y2)
        labels = [label]  # Asignar la etiqueta de la vista

        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)

        return frame, target


# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Dividir el DataFrame en conjunto de entrenamiento y validación
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Crear Datasets de entrenamiento y validación
train_dataset = EchoDataset(train_df, dataset_path, masks_path, transform=transform)
val_dataset = EchoDataset(val_df, dataset_path, masks_path, transform=transform)

# DataLoaders
batch_size = 4
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Modelo preentrenado (Faster-RCNN)
model = models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')

# Modificar el clasificador del modelo para el número de clases (1 objeto más fondo)
num_classes = 4  # 3 classes (psax_aov, plax, 3c) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.rpn.nms_thresh = 0.7  # Mantén el valor por defecto (0.7) o ajusta a 0.5 si es necesario
model.rpn.post_nms_top_n_test = 1  # Solo guarda una caja propuesta en la fase de inferencia
model.rpn.post_nms_top_n_train = 1  # Solo guarda una caja propuesta en la fase de entrenamiento
model.roi_heads.detections_per_img = 1  # Limita a una predicción por imagen

# Configuración de optimizador y scheduler (Cosine Annealing con Warm Restarts)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Función para calcular IoU (Intersection over Union)
def calculate_iou(pred_boxes, true_boxes):
    return box_iou(pred_boxes, true_boxes)

# Función para calcular la precisión de los matches por IoU
def calculate_match_precision(pred_boxes, true_boxes, iou_threshold=0.5):
    iou = calculate_iou(pred_boxes, true_boxes)
    matches = iou > iou_threshold  # Verdaderos positivos según el IoU
    match_true_positives = matches.sum().item()  # Contar verdaderos positivos
    total_pred = len(pred_boxes)
    
    match_precision = match_true_positives / total_pred if total_pred > 0 else 0
    return match_precision

# Función para calcular la precisión de las etiquetas (labels)
def calculate_label_precision(pred_labels, true_labels):
    correct_labels = (pred_labels == true_labels).sum().item()
    total_pred = len(pred_labels)
    
    label_precision = correct_labels / total_pred if total_pred > 0 else 0
    return label_precision

# Función de validación con cálculo de mAP
def validate_model_with_map(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    match_precisions = []
    label_precisions = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validación con precisión separada"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Hacer predicciones
            outputs = model(images)
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu()
                pred_labels = output['labels'].cpu()
                true_boxes = targets[i]['boxes'].cpu()
                true_labels = targets[i]['labels'].cpu()

                # Calcular precisión de los matches (IoU > umbral)
                match_precision = calculate_match_precision(pred_boxes, true_boxes, iou_threshold)
                match_precisions.append(match_precision)

                # Calcular precisión de las etiquetas (labels)
                label_precision = calculate_label_precision(pred_labels, true_labels)
                label_precisions.append(label_precision)

    # Calcular la precisión promedio en matches y labels
    avg_match_precision = sum(match_precisions) / len(match_precisions) if len(match_precisions) > 0 else 0
    avg_label_precision = sum(label_precisions) / len(label_precisions) if len(label_precisions) > 0 else 0

    # Mostrar los resultados
    print(f"Precisión de matches (IoU > {iou_threshold}): {avg_match_precision:.4f}")
    print(f"Precisión de las etiquetas: {avg_label_precision:.4f}")
    
    return avg_match_precision, avg_label_precision

# Función para guardar el modelo
def save_model(model, epoch, prefix):
    model_path = f"{prefix}/model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")

# Función para guardar los resultados en archivo de texto
def save_training_info(results_dir, epoch, train_loss, avg_match_precision, avg_label_precision):
    info_file = os.path.join(results_dir, f'{results_dir.split(os.sep)[-1]}_info.txt')
    with open(info_file, 'a') as f:
        f.write(f"Epoch {epoch + 1}:\n")
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Match Precision: {avg_match_precision:.4f}\n")
        f.write(f"Label Precision: {avg_label_precision:.4f}\n")
        f.write("="*50 + "\n")

# Función para crear el directorio de resultados
def create_results_directory(base_path='results'):
    current_time = time.strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_path, current_time)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Función de entrenamiento con visualización y guardado de modelo
def train_model_with_visualization(model, train_loader, val_loader, optimizer, scheduler=None, num_epochs=30):
    best_val_loss = 0.0  # Para rastrear el mejor modelo

    # Crear el directorio de resultados
    results_dir = create_results_directory()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # tqdm para mostrar progreso en el entrenamiento
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass y cálculo de pérdidas
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Actualizar el learning rate usando el scheduler de Cosine Annealing
        if scheduler != None:
            scheduler.step()

        # Total de imágenes en entrenamiento
        total_images_train = len(train_loader.dataset)

        # Calcular la pérdida promedio por imagen en entrenamiento
        avg_train_loss = train_loss / total_images_train

        # Validar el modelo y calcular el mAP en la validación
        avg_match_precision, avg_label_precision = validate_model_with_map(model, val_loader, device)

        # Mostrar el progreso
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss (per image): {avg_train_loss:.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Precisión de matches: {avg_match_precision:.4f}, Precisión de labels: {avg_label_precision:.4f}')

        # Guardar el modelo después de cada época
        save_model(model, epoch, results_dir)

        # Guardar la información de cada epoch en el archivo de texto
        save_training_info(results_dir, epoch, avg_train_loss, avg_match_precision, avg_label_precision)

        # Guardar el mejor modelo (basado en la precisión de matches o labels, lo que prefieras)
        if avg_match_precision > best_val_loss:  # O usa avg_label_precision si prefieres
            best_val_loss = avg_match_precision
            save_model(model, "best", results_dir)

# Entrenar el modelo con visualización de resultados
train_model_with_visualization(model, train_data_loader, val_data_loader, optimizer, scheduler=None, num_epochs=30)
