import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import cv2

# Función para aplicar zoom central
def zoom_central(image_array, zoom_factor=1.5):
    h, w = image_array.shape[:2]
    crop_h = int(h / zoom_factor)
    crop_w = int(w / zoom_factor)
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    cropped_image = image_array[start_y:start_y+crop_h, start_x:start_x+crop_w]
    return cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_AREA)

# Función que procesa cada fila
def process_row(row):
    try:
        # Extract p1 and p2 coordinates
        p1 = row['p1_new']
        p2 = row['p2_new']

        # Determine the view
        if row['data'] == 'data_psax':
            base_dir = r'\\NAS3_Z\all\BKP_PERE\ARQPSAX\data\frames_croped'
        elif row['data'] == 'data_plax':
            base_dir = r'\\NAS3_Z\all\BKP_PERE\ARQPSAX\data_plax\frames_croped_plax'
        elif row['data'] == 'data_3ch':
            base_dir = r'\\NAS3_Z\all\BKP_PERE\ARQPSAX\data_3ch\frames_croped_3ch'

        # Build the path to the frames directory
        dicom = row['echo']
        subfolder = row['subcarpeta']
        patient_folder = row['carpeta paciente']

        frame_path = os.path.join(base_dir, '128_128', subfolder, patient_folder, dicom)
        crop_dir = r'\\NAS3_Z\all\BKP_PERE\valveDetection'
        # Check if the path exists
        if os.path.exists(frame_path):
            for f in os.listdir(frame_path):
                # Create the path to save the resized image
                if row['data'] == 'data_psax':
                    # save_path = f'data/frames_valve/256_256/{subfolder}/{patient_folder}/{dicom}'
                    save_path = f'{crop_dir}/data/frames_valve/256_256_new/{subfolder}/{patient_folder}/{dicom}'
                elif row['data'] == 'data_plax':
                    # save_path = f'data_plax/frames_valve/256_256/{subfolder}/{patient_folder}/{dicom}'
                    save_path = f'{crop_dir}/data_plax/frames_valve/256_256_new/{subfolder}/{patient_folder}/{dicom}'
                elif row['data'] == 'data_3ch':
                    # save_path = f'data_3ch/frames_valve/256_256/{subfolder}/{patient_folder}/{dicom}'
                    save_path = f'{crop_dir}/data_3ch/frames_valve/256_256_new/{subfolder}/{patient_folder}/{dicom}'

                save_file = os.path.join(save_path, f)
                if os.path.exists(save_file):
                    # print(f"File already exists, skipping: {save_file}")
                    continue

                frame_file = os.path.join(frame_path, f)
                frame = np.load(frame_file)

                # If p1 and p2 are (-1, -1), apply a central zoom
                if p1 == (-1, -1) and p2 == (-1, -1):
                    image_resized = zoom_central(frame, zoom_factor=1.5)
                else:
                    x_min, x_max = sorted([p1[0], p2[0]])  # Ensure coordinates are ordered correctly
                    y_min, y_max = sorted([p1[1], p2[1]])
                    cropped_frame = frame[y_min:y_max, x_min:x_max]  # Crop the frame
                    image_resized = cv2.resize(cropped_frame, (256, 256), interpolation=cv2.INTER_AREA)

                # Ensure the save directory exists
                os.makedirs(save_path, exist_ok=True)

                # Save the resized frame as a .npy file
                np.save(save_file, np.array(image_resized))

                print(f"Frame saved at {save_file}")
        else:
            print(f"Frame path does not exist: {frame_path}")
    except Exception as e:
        print(f"Error processing row: {row['echo']} - {e}")

# Main function to parallelize the process
def process_dataframe(df, num_threads=32):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions if they occurred during the processing

if __name__ == "__main__":
    # Load the CSV into a DataFrame
    # df = pd.read_csv('files/inference_output_12_24m_corrected_upd_3_256.csv') 
    df = pd.read_csv('all_avcs_corrected_def.csv') 

    # Convert p1_new and p2_new from strings to tuples
    df['p1_new'] = df['p1_new'].apply(ast.literal_eval)
    df['p2_new'] = df['p2_new'].apply(ast.literal_eval)

    print('Empezando a procesar...')
    # Run the process with 4 threads (you can adjust the number of threads as needed)
    process_dataframe(df, num_threads=32)
