import os
import torch
import numpy as np
import pandas as pd
from torchvision import models
from PIL import Image
from torchvision.transforms import ToTensor
import csv
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ruta del modelo entrenado
model_path = 'results/20241014_124322/model_epoch_best.pth'

# Rutas de las imágenes
base_dirs = [r'\\NAS3_Z\all\BKP_PERE\ARQPSAX\data\frames_croped\128_128', 
             r'\\NAS3_Z\all\BKP_PERE\ARQPSAX\data_plax\frames_croped_plax\128_128',
             r'\\NAS3_Z\all\BKP_PERE\ARQPSAX\data_3ch\frames_croped_3ch\128_128']


# Subcarpetas
# subfolders = ['48m'] #['12m', '24m', 'basal', 'final']
# subfolders = ['12m', '24m']
subfolders = ['basal', 'final']

# CSV para guardar los resultados
output_csv = 'inference_new_c9.csv'

# Etiquetas de las vistas (ajustar si es necesario)
label_to_view = {1: 'psax_aov', 2: 'plax', 3: '3c'}

# Cargar el modelo
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1')

num_classes = 4  # 3 clases (psax_aov, plax, 3c) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.rpn.nms_thresh = 0.7  # Mantén el valor por defecto (0.7) o ajusta a 0.5 si es necesario
model.rpn.post_nms_top_n_test = 1  # Solo guarda una caja propuesta en la fase de inferencia
model.rpn.post_nms_top_n_train = 1  # Solo guarda una caja propuesta en la fase de entrenamiento
model.roi_heads.detections_per_img = 1  # Limita a una predicción por imagen

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transformación de imagen
transform = ToTensor()

# Función para realizar inferencia
def inference_on_image(image_path):
    # Cargar la imagen (npy)
    image_np = np.load(image_path)
    image = Image.fromarray(image_np).resize((256, 256))
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Realizar predicción
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extraer la caja predicha y la etiqueta
    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].cpu().numpy()

    # Verificar que haya al menos una predicción
    if len(pred_boxes) > 0:
        # Usar la predicción con el puntaje más alto
        max_idx = np.argmax(pred_scores)
        p1 = pred_boxes[max_idx][:2]  # (x_min, y_min)
        p2 = pred_boxes[max_idx][2:]  # (x_max, y_max)
        label = pred_labels[max_idx]
        vista = label_to_view.get(label, 'unknown')
        return p1, p2, vista
    else:
        return None, None, None

# Leer el CSV existente si ya ha sido creado
if os.path.exists(output_csv):
    df_results = pd.read_csv(output_csv)
else:
    df_results = pd.DataFrame(columns=['subcarpeta', 'carpeta paciente', 'echo', 'frame', 'p1', 'p2', 'vista', 'data'])

# Abrir el CSV para escribir los nuevos resultados
with open(output_csv, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Escribir el encabezado solo si el archivo está vacío
    if df_results.empty:
        writer.writerow(['subcarpeta', 'carpeta paciente', 'echo', 'frame', 'p1', 'p2', 'vista', 'data'])

    # Convertir el DataFrame a un set para fácil comparación
    processed_frames = set(df_results.apply(lambda row: f"{row['subcarpeta']}/{row['carpeta paciente']}/{row['echo']}/{row['frame']}", axis=1))

    # Recorrer las subcarpetas y realizar inferencia
    for base_dir in base_dirs:
        if base_dir == r'\\NAS3_Z\all\BKP_PERE\ARQPSAX\data\frames_croped\128_128':
            data = 'data_psax'
        elif base_dir == r'\\NAS3_Z\all\BKP_PERE\ARQPSAX\data_plax\frames_croped_plax\128_128':
            data = 'data_plax'
        elif base_dir == r'\\NAS3_Z\all\BKP_PERE\ARQPSAX\data_3ch\frames_croped_3ch\128_128':
            data = 'data_3ch'
        for subfolder in subfolders:
            subfolder_path = os.path.join(base_dir, subfolder)
            for patient_folder in os.listdir(subfolder_path):
                patient_path = os.path.join(subfolder_path, patient_folder)
                for dicom_folder in os.listdir(patient_path):
                    dicom_path = os.path.join(patient_path, dicom_folder)
                    if os.path.isdir(dicom_path):  # Verificar si es una carpeta
                        for frame_file in tqdm(os.listdir(dicom_path)):
                            if frame_file.endswith('.npy'):  # Asumimos que las imágenes son archivos .npy
                                frame_path = os.path.join(dicom_path, frame_file)
                                frame_id = f"{subfolder}/{patient_folder}/{dicom_folder}/{os.path.splitext(frame_file)[0]}"

                                # Si el frame ya está procesado, saltar
                                if frame_id in processed_frames:
                                    continue
                                
                                # Realizar inferencia
                                try:
                                    p1, p2, vista = inference_on_image(frame_path)
                                except Exception as e:
                                    print(f'Error en {frame_path}: {e}')
                                    p1, p2, vista = None, None, None
                                # Si se obtuvo una predicción válida, guardarla en el CSV
                                if p1 is not None and p2 is not None:
                                    frame_number = os.path.splitext(frame_file)[0]  # Remover la extensión del archivo
                                    writer.writerow([subfolder, patient_folder, dicom_folder, frame_number, list(p1), list(p2), vista, data])
                                else:
                                    print(f'error {p1, p2, vista}')
                        
                        # Imprimir mensaje cuando se ha terminado de procesar un dicom_folder
                        print(f"Finalizado DICOM: {dicom_folder} en {patient_path}")

print(f"Inferencia completada. Resultados guardados en {output_csv}")
