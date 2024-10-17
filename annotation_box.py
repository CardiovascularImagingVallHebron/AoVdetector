import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# Cargar el CSV
csv_path = r'E:\25366074H\Documents\echoqual\data\tartaglia_db_beta_modified.csv'
df = pd.read_csv(csv_path, delimiter=';')

# Archivo donde guardar las anotaciones
output_csv = 'anotaciones_new.csv'

# Si el archivo de anotaciones no existe, crearlo con el encabezado
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['view', 'echo_id', 'frame', '1er clic (x, y)', '2o clic (x, y)'])

# Archivo donde guardar los echo_id descartados
discarded_csv = 'discarded_echo_ids.csv'

# Si el archivo de descartes no existe, crearlo
if not os.path.exists(discarded_csv):
    with open(discarded_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['echo_id'])

# Cargar el archivo de anotaciones existentes para verificar duplicados
annotated_data = pd.read_csv(output_csv)

# Cargar el archivo de echo_id descartados
discarded_data = pd.read_csv(discarded_csv)

# Filtrar por 'view_annotation'
valid_views = ['3c', 'plax', 'psax_aov', 'psax']
filtered_df = df[df['view_annotation'].isin(valid_views)]

# Ruta base donde están las carpetas
dataset_path = r'\\NAS3_Z\all\BKP_PERE\echoqual\data\frames_resized\256_256_new'

# Variables globales para controlar los frames y el estado
current_echo_id = None
npy_files = []
current_file_index = 0
first_click = None  # Primer clic de la caja (x, y)
second_click = None  # Segundo clic de la caja (x, y)
drawing_box = False  # Indicador de si estamos dibujando la caja
view = None  # Variable global para almacenar el 'view'
echo_folder = None  # Variable global para almacenar el directorio del echo_id
current_index = 0  # Índice global para rastrear el progreso en filtered_df

# Función para verificar si ya se ha anotado el view y echo_id
def is_echo_annotated(view, echo_id):
    global annotated_data
    existing_annotations = annotated_data[
        (annotated_data['view'] == view) & 
        (annotated_data['echo_id'] == echo_id)
    ]
    return not existing_annotations.empty

# Función para verificar si el echo_id ha sido descartado
def is_echo_discarded(echo_id):
    global discarded_data
    return not discarded_data[discarded_data['echo_id'] == echo_id].empty

# Cargar el CSV de anotaciones para las cajas p1, p2
anotaciones_csv_path = r'data/anotaciones.csv'
anotaciones_df = pd.read_csv(anotaciones_csv_path)

# Ruta base para el segundo frame
frames_dataset_path = r'\\NAS3_Z\all\BKP_PERE\echoqual\data\frames'

# Función para dibujar el segundo gráfico
def show_second_frame_with_box():
    global current_file_index, npy_files, current_echo_id, anotaciones_df, frames_dataset_path

    # Filtrar el dataframe de anotaciones para obtener la fila correspondiente
    annotation_row = anotaciones_df[
        (anotaciones_df['echo_id'] == current_echo_id)
    ]

    if not annotation_row.empty:
        # Extraer las coordenadas de la caja p1 y p2
        p1 = eval(annotation_row.iloc[0]['1er clic (x, y)'])
        p2 = eval(annotation_row.iloc[0]['2o clic (x, y)'])
        # Cargar el frame correspondiente del segundo dataset
        frame_file = f"frame_{annotation_row.iloc[0]['frame']:04d}.npy"
        frame_path = os.path.join(frames_dataset_path, current_echo_id, frame_file)

        if os.path.exists(frame_path):
            second_frame = np.load(frame_path)

            if second_frame.ndim == 2:
                plt.subplot(1, 2, 2)  # Crear el segundo subplot a la derecha
                plt.imshow(second_frame, cmap='gray')
                frame_name = annotation_row.iloc[0]['frame']
                plt.title(f'Segundo Frame - Echo ID: {current_echo_id}, Frame: {frame_name}')
                plt.axis('off')

                # Dibujar la caja en el segundo frame
                plt.gca().add_patch(plt.Rectangle(p1, p2[0] - p1[0], p2[1] - p1[1], 
                                                  fill=False, edgecolor='red', linewidth=2))
                plt.draw()
            else:
                print(f'El frame de {frame_path} no tiene el formato correcto para ser ploteado (debería ser 2D)')
        else:
            print(f'No se encontró el frame en la ruta: {frame_path}')
    else:
        print(f'No se encontraron anotaciones para echo_id {current_echo_id} y frame {current_file_index}')


# Modificación de la función show_frame para incluir el segundo frame con la caja
def show_frame():
    global current_file_index, npy_files, first_click, second_click, drawing_box, echo_folder
    
    if npy_files:
        file_path = os.path.join(echo_folder, npy_files[current_file_index])
        frame = np.load(file_path)

        if frame.ndim == 2:
            show_second_frame_with_box()
            plt.subplot(1, 2, 1)  # Crear el primer subplot a la izquierda
            plt.imshow(frame, cmap='gray')
            plt.title(f'Echo ID: {current_echo_id}, Archivo: {npy_files[current_file_index]}, Vista: {view}')
            plt.axis('off')
            plt.draw()

            # Mostrar el segundo frame con la caja
        else:
            print(f'El frame de {npy_files[current_file_index]} no tiene el formato correcto para ser ploteado (debería ser 2D)')
    else:
        print(f'No hay archivos .npy disponibles para mostrar en {current_echo_id}')

# Evento de pulsación de tecla para avanzar, retroceder, descartar o cancelar el primer clic
def on_key(event):
    global current_file_index, first_click, second_click, drawing_box, current_echo_id, current_index
    
    if event.key == 'right':  # Avanzar al siguiente frame con la flecha derecha
        current_file_index = (current_file_index + 1) % len(npy_files)
        plt.clf()  # Limpiar la figura actual
        show_frame()
        
    elif event.key == 'left':  # Retroceder al frame anterior con la flecha izquierda
        current_file_index = (current_file_index - 1) % len(npy_files)
        plt.clf()  # Limpiar la figura actual
        show_frame()
    
    elif event.key == 'd':  # Si se pulsa la tecla 'x', descartar el echo y avanzar al siguiente
        print(f'Echo ID {current_echo_id} descartado, avanzando al siguiente echo.')

        # Guardar el echo_id descartado en el CSV de descartados
        with open(discarded_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_echo_id])
        
        # Actualizar el dataframe de descartados
        new_row = pd.DataFrame({'echo_id': [current_echo_id]})
        global discarded_data
        discarded_data = pd.concat([discarded_data, new_row], ignore_index=True)
        
        # Avanzar al siguiente echo directamente
        plt.close()  # Cerrar la ventana actual para evitar problemas visuales
        advance_to_next_echo()  # Llamar a la función para avanzar
    
    elif event.key == 'escape' and first_click is not None:  # Si se pulsa Escape, restablecer el primer clic
        print('Primer clic cancelado.')
        first_click = None
        drawing_box = False
        plt.clf()  # Limpiar la figura actual
        show_frame()

# Evento de clic del mouse para capturar las coordenadas y dibujar la caja
def on_click(event):
    global first_click, second_click, drawing_box, annotated_data
    
    if event.inaxes:  # Si el clic está dentro de la imagen
        if first_click is None:
            # Primer clic (inicio de la caja)
            first_click = (event.xdata, event.ydata)
            drawing_box = True
            print(f'Primer clic: {first_click}')
        else:
            # Segundo clic (final de la caja)
            second_click = (event.xdata, event.ydata)
            print(f'Segundo clic: {second_click}')
            
            # Guardar la anotación en el archivo CSV
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([view, current_echo_id, current_file_index, first_click, second_click])
            
            # Actualizar el dataframe de anotaciones
            new_row = pd.DataFrame({
                'view': [view], 'echo_id': [current_echo_id], 
                'frame': [current_file_index], '1er clic (x, y)': [first_click], '2o clic (x, y)': [second_click]
            })
            annotated_data = pd.concat([annotated_data, new_row], ignore_index=True)
            
            print(f'Anotación guardada: {first_click} a {second_click} en frame {current_file_index}')

            # Reset para la siguiente anotación
            first_click = None
            second_click = None
            drawing_box = False

            # Avanzar al siguiente echo después de la anotación
            advance_to_next_echo()

# Función para avanzar al siguiente 'echo_id'
def advance_to_next_echo():
    global current_file_index, npy_files, current_echo_id, echo_folder, view, current_index
    # Reiniciar los clics y el estado
    first_click = None
    second_click = None
    drawing_box = False
    
    # Avanzar al siguiente echo_id
    current_file_index = 0  # Reiniciar al primer archivo del siguiente echo
    plt.clf()  # Limpiar la figura actual
    plt.close()  # Cerrar la ventana actual para pasar al siguiente echo

    # Calcular cuántos echo_id quedan por anotar
    # Obtener los echo_id que ya han sido anotados o descartados
    annotated_echo_ids = set(annotated_data['echo_id'].unique())
    discarded_echo_ids = set(discarded_data['echo_id'].unique())
    
    # Conjuntos únicos de echo_id anotados o descartados
    processed_echo_ids = annotated_echo_ids.union(discarded_echo_ids)
    
    # Restar los echo_id procesados (anotados o descartados) del dataframe filtrado
    remaining_to_annotate = len(filtered_df) - filtered_df['echo_id'].isin(processed_echo_ids).sum()

    print(f"Quedan {remaining_to_annotate} echo_id por anotar")

    # Incrementar el índice global para avanzar al siguiente echo_id
    while current_index < len(filtered_df):
        row = filtered_df.iloc[current_index]
        current_echo_id = row['echo_id']
        view = row['view_annotation']
        
        # Verificar si ya se ha anotado el echo_id y view o si ha sido descartado
        if is_echo_annotated(view, current_echo_id):
            print(f'El echo {current_echo_id} con la vista {view} ya ha sido anotado. Saltando.')
            current_index += 1  # Saltar al siguiente
            continue  # Saltar a la siguiente iteración si ya fue anotado

        if is_echo_discarded(current_echo_id):
            print(f'El echo {current_echo_id} ha sido descartado. Saltando.')
            current_index += 1  # Saltar al siguiente
            continue  # Saltar si ya fue descartado

        echo_folder = os.path.join(dataset_path, current_echo_id)
        
        if os.path.isdir(echo_folder):
            # Obtener todos los archivos .npy de la carpeta
            npy_files = [f for f in os.listdir(echo_folder) if f.endswith('.npy')]
            
            if npy_files:
                # Mostrar el primer frame (índice 0)
                current_file_index = 0
                plt.figure(figsize=(10,8))
                show_frame()
                
                # Conectar los eventos de teclado, clic y movimiento del mouse
                plt.gcf().canvas.mpl_connect('key_press_event', on_key)
                plt.gcf().canvas.mpl_connect('button_press_event', on_click)
                plt.gcf().canvas.mpl_connect('motion_notify_event', on_motion)
                
                plt.show()  # Mostrar el gráfico y esperar interacción del usuario
                current_index += 1  # Avanzar al siguiente echo_id después de procesar este
                break
            else:
                print(f'No se encontraron archivos .npy en la carpeta de {current_echo_id}')
        else:
            print(f'La carpeta para {current_echo_id} no existe en el dataset')
        current_index += 1
# Evento de movimiento del mouse para dibujar la caja interactiva
def on_motion(event):
    global first_click, drawing_box
    
    if drawing_box and first_click is not None and event.inaxes:
        plt.clf()  # Limpiar la imagen para actualizar el dibujo de la caja
        show_frame()  # Mostrar nuevamente el frame actual
        
        # Dibujar la caja temporal entre el primer clic y la posición actual del cursor
        current_x, current_y = event.xdata, event.ydata
        plt.gca().add_patch(plt.Rectangle(first_click, current_x - first_click[0], current_y - first_click[1], 
                                          fill=False, edgecolor='red', linewidth=2))
        plt.draw()

# Iniciar el proceso con el primer 'echo_id'
advance_to_next_echo()
