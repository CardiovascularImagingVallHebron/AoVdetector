import os
import numpy as np
import cv2
import pandas as pd

def img_crop_v2(img: np.ndarray, target_height: int, target_width: int, point: tuple = None) -> np.ndarray:
    """
        Función que recibe una imagen (frame) de entrada y que corta o añade bordes (padding) 
        en función del tamaño de altura y ancho objetivo, el punto es opcional por si se quiere
        informar que punto de referencia tomar para el padding.
    """    

    current_height, current_width = img.shape[:2]

    # Relleno (Padding)
    pad_width = max(0, target_width - current_width)
    pad_height = max(0, target_height - current_height)

    # Distribuir el relleno uniformemente entre ambos lados
    pad_width_left = pad_width // 2
    pad_width_right = pad_width - pad_width_left
    pad_height_top = pad_height // 2
    pad_height_bottom = pad_height - pad_height_top

    padded_img = np.pad(img, ((pad_height_top, pad_height_bottom), 
                              (pad_width_left, pad_width_right)),
                        'constant', constant_values=0)

    # Si el punto no se proporciona, usa el centro
    if point is None:
        point = (padded_img.shape[1] // 2, padded_img.shape[0] // 2)

    # Calcular los puntos de inicio y finalización para el recorte
    start_x = max(0, point[0] - target_width // 2)
    end_x = start_x + target_width
    start_y = max(0, point[1] - target_height // 2)
    end_y = start_y + target_height

    # Ajustar los puntos de recorte para evitar salirse de los límites de la imagen
    start_x = min(max(0, start_x), padded_img.shape[1] - target_width)
    start_y = min(max(0, start_y), padded_img.shape[0] - target_height)
    end_x = start_x + target_width
    end_y = start_y + target_height

    cropped_img = padded_img[start_y:end_y, start_x:end_x]

    return cropped_img

# Modificar la función para aceptar también los puntos y devolver las coordenadas redimensionadas
def new_resize_img(pixel_array, mask_array, width, height, p1=None, p2=None, new_p1=None, new_p2=None):
    # Calculos de la máscara
    mask_array = np.array(mask_array, dtype='uint8')
    row, cols = np.where(mask_array > 0)
    original_dist_x = np.max(cols) - np.min(cols)
    original_dist_y = np.max(row) - np.min(row)
    center_x = round(original_dist_x / 2) + np.min(cols)
    center_y = round(original_dist_y / 2) + np.min(row)

    # Calcular el desplazamiento del centro de la imagen
    offset_x = center_x - (original_dist_x // 2)
    offset_y = center_y - (original_dist_y // 2)

    # Redimensión de la imagen y la máscara
    if original_dist_x > original_dist_y:
        new_image = img_crop_v2(pixel_array, original_dist_x + 6, original_dist_x + 6, point=(center_x, center_y))
        new_mask = img_crop_v2(mask_array, original_dist_x + 6, original_dist_x + 6, point=(center_x, center_y))
    else:
        new_image = img_crop_v2(pixel_array, original_dist_y + 6, original_dist_y + 6, point=(center_x, center_y))
        new_mask = img_crop_v2(mask_array, original_dist_y + 6, original_dist_y + 6, point=(center_x, center_y))

    # Escalar la imagen y la máscara
    resized_image = cv2.resize(new_image, (width, height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(new_mask, (width, height), interpolation=cv2.INTER_AREA)

    # Convertir a float32
    resized_image = resized_image.astype(np.float32)
    resized_mask = resized_mask.astype(np.float32)

    # Ajustar puntos al nuevo tamaño
    # Los puntos deben restarse con el offset para obtener la posición relativa después del recorte
    if p1!=None:
        p1_relative = (p1[0] - offset_x, p1[1] - offset_y)
        p2_relative = (p2[0] - offset_x, p2[1] - offset_y)
        new_p1_relative = (new_p1[0] - offset_x, new_p1[1] - offset_y)
        new_p2_relative = (new_p2[0] - offset_x, new_p2[1] - offset_y)

        # Escalar los puntos en base al nuevo tamaño
        scale_x = width / new_image.shape[1]
        scale_y = height / new_image.shape[0]

        p1_resized = (p1_relative[0] * scale_x, p1_relative[1] * scale_y)
        p2_resized = (p2_relative[0] * scale_x, p2_relative[1] * scale_y)
        new_p1_resized = (new_p1_relative[0] * scale_x, new_p1_relative[1] * scale_y)
        new_p2_resized = (new_p2_relative[0] * scale_x, new_p2_relative[1] * scale_y)
    else:
        p1_resized = 0
        p2_resized = 0
        new_p1_resized = 0
        new_p2_resized = 0
    


    return resized_image, resized_mask, None, p1_resized, p2_resized, new_p1_resized, new_p2_resized

