import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

# Load CSV
csv_path = 'data\example.csv'
df = pd.read_csv(csv_path, delimiter=';')

# File to save annotations
output_csv = 'anotaciones_new.csv'

# If the annotations file does not exist, create it with the header
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['view', 'echo_id', 'frame', '1er clic (x, y)', '2o clic (x, y)'])

# File to save discarded echo_ids
discarded_csv = 'discarded_echo_ids.csv'

# If the discarded file does not exist, create it
if not os.path.exists(discarded_csv):
    with open(discarded_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['echo_id'])

# Load the existing annotations file to check for duplicates
annotated_data = pd.read_csv(output_csv)

# Load the discarded echo_id file
discarded_data = pd.read_csv(discarded_csv)

# Filter by 'view_annotation'
valid_views = ['3c', 'plax', 'psax_aov', 'psax']
filtered_df = df[df['view_annotation'].isin(valid_views)]

# Base path where the folders are located
dataset_path = r'your_path'

# Global variables to control frames and state
current_echo_id = None
npy_files = []
current_file_index = 0
first_click = None  # First click of the box (x, y)
second_click = None  # Second click of the box (x, y)
drawing_box = False  # Indicator if we are drawing the box
view = None  # Global variable to store the 'view'
echo_folder = None  # Global variable to store the echo_id directory
current_index = 0  # Global index to track progress in filtered_df

# Function to check if the view and echo_id have already been annotated
def is_echo_annotated(view, echo_id):
    global annotated_data
    existing_annotations = annotated_data[
        (annotated_data['view'] == view) & 
        (annotated_data['echo_id'] == echo_id)
    ]
    return not existing_annotations.empty

# Function to check if the echo_id has been discarded
def is_echo_discarded(echo_id):
    global discarded_data
    return not discarded_data[discarded_data['echo_id'] == echo_id].empty

# Load the CSV of annotations for boxes p1, p2
anotaciones_csv_path = r'data/anotaciones.csv'
anotaciones_df = pd.read_csv(anotaciones_csv_path)

# Base path for the second frame
frames_dataset_path = r'frame_path'

# Function to draw the second plot
def show_second_frame_with_box():
    global current_file_index, npy_files, current_echo_id, anotaciones_df, frames_dataset_path

    # Filter the annotations dataframe to get the corresponding row
    annotation_row = anotaciones_df[
        (anotaciones_df['echo_id'] == current_echo_id)
    ]

    if not annotation_row.empty:
        # Extract the coordinates of the box p1 and p2
        p1 = eval(annotation_row.iloc[0]['1er clic (x, y)'])
        p2 = eval(annotation_row.iloc[0]['2o clic (x, y)'])
        # Load the corresponding frame from the second dataset
        frame_file = f"frame_{annotation_row.iloc[0]['frame']:04d}.npy"
        frame_path = os.path.join(frames_dataset_path, current_echo_id, frame_file)

        if os.path.exists(frame_path):
            second_frame = np.load(frame_path)

            if second_frame.ndim == 2:
                plt.subplot(1, 2, 2)  
                plt.imshow(second_frame, cmap='gray')
                frame_name = annotation_row.iloc[0]['frame']
                plt.title(f'Second Frame - Echo ID: {current_echo_id}, Frame: {frame_name}')
                plt.axis('off')

                plt.gca().add_patch(plt.Rectangle(p1, p2[0] - p1[0], p2[1] - p1[1], 
                                                  fill=False, edgecolor='red', linewidth=2))
                plt.draw()
            else:
                print(f'The frame at {frame_path} does not have the correct format to be plotted (should be 2D)')
        else:
            print(f'Frame not found at path: {frame_path}')
    else:
        print(f'No annotations found for echo_id {current_echo_id} and frame {current_file_index}')


# Modification of the show_frame function to include the second frame with the box
def show_frame():
    global current_file_index, npy_files, first_click, second_click, drawing_box, echo_folder
    
    if npy_files:
        file_path = os.path.join(echo_folder, npy_files[current_file_index])
        frame = np.load(file_path)

        if frame.ndim == 2:
            show_second_frame_with_box()
            plt.subplot(1, 2, 1)  
            plt.imshow(frame, cmap='gray')
            plt.title(f'Echo ID: {current_echo_id}, Archivo: {npy_files[current_file_index]}, Vista: {view}')
            plt.axis('off')
            plt.draw()

            # Show the second frame with the box
        else:
            print(f'The frame of {npy_files[current_file_index]} does not have the correct format to be plotted (should be 2D)')
    else:
        print(f'No .npy files available to display for {current_echo_id}')

# Key press event to advance, go back, discard, or cancel the first click
def on_key(event):
    global current_file_index, first_click, second_click, drawing_box, current_echo_id, current_index
    
    if event.key == 'right':  # Advance to the next frame with the right arrow
        current_file_index = (current_file_index + 1) % len(npy_files)
        plt.clf()  # Clear the current figure
        show_frame()
        
    elif event.key == 'left':  # Go back to the previous frame with the left arrow
        current_file_index = (current_file_index - 1) % len(npy_files)
        plt.clf()  # Clear the current figure
        show_frame()
    
    elif event.key == 'd':  # If the 'd' key is pressed, discard the echo and move to the next one
        print(f'Echo ID {current_echo_id} discarded, moving to the next echo.')
        # Save the discarded echo_id to the discarded CSV
        with open(discarded_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_echo_id])
        
        # Update the discarded dataframe
        new_row = pd.DataFrame({'echo_id': [current_echo_id]})
        global discarded_data
        discarded_data = pd.concat([discarded_data, new_row], ignore_index=True)
        
        # Advance to the next echo directly
        plt.close()  # Close the current window to avoid visual issues
        advance_to_next_echo()  # Call the function to advance
    
    elif event.key == 'escape' and first_click is not None:  # If Escape is pressed, reset the first click
        print('First click canceled.')
        first_click = None
        drawing_box = False
        plt.clf()  # Clear the current figure   
        show_frame()

# Mouse click event to capture coordinates and draw the box
def on_click(event):
    global first_click, second_click, drawing_box, annotated_data
    
    if event.inaxes:  # If the click is inside the image
        if first_click is None:
            # First click (start of the box)
            first_click = (event.xdata, event.ydata)
            drawing_box = True
            print(f'First click: {first_click}')
        else:
            # Second click (end of the box)
            second_click = (event.xdata, event.ydata)
            print(f'Second click: {second_click}')
            
            # Save the annotation to the CSV file
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([view, current_echo_id, current_file_index, first_click, second_click])
            
            # Update the annotated dataframe
            new_row = pd.DataFrame({
                'view': [view], 'echo_id': [current_echo_id], 
                'frame': [current_file_index], '1er clic (x, y)': [first_click], '2o clic (x, y)': [second_click]
            })
            annotated_data = pd.concat([annotated_data, new_row], ignore_index=True)
            
            print(f'Saved annotation: {first_click} to {second_click} on frame {current_file_index}')

            # Reset for the next annotation
            first_click = None
            second_click = None
            drawing_box = False

            # Advance to the next echo after annotation
            advance_to_next_echo()

# Function to advance to the next 'echo_id'
def advance_to_next_echo():
    global current_file_index, npy_files, current_echo_id, echo_folder, view, current_index
    # Reset clicks and state
    first_click = None
    second_click = None
    drawing_box = False
    
    # Advance to the next echo_id
    current_file_index = 0  # Reset to the first file of the next echo
    plt.clf()  # Clear the current figure
    plt.close()  # Close the current window to move to the next echo

    # Calculate how many echo_id remain to be annotated
    # Get the echo_id that have already been annotated or discarded
    annotated_echo_ids = set(annotated_data['echo_id'].unique())
    discarded_echo_ids = set(discarded_data['echo_id'].unique())
    
    # Unique sets of annotated or discarded echo_id
    processed_echo_ids = annotated_echo_ids.union(discarded_echo_ids)
    
    # Subtract the processed echo_id (annotated or discarded) from the filtered dataframe
    remaining_to_annotate = len(filtered_df) - filtered_df['echo_id'].isin(processed_echo_ids).sum()

    print(f"{remaining_to_annotate} echo_id remain to be annotated")

    # Increment the global index to move to the next echo_id
    while current_index < len(filtered_df):
        row = filtered_df.iloc[current_index]
        current_echo_id = row['echo_id']
        view = row['view_annotation']
        
        # Check if the echo_id and view have already been annotated or discarded
        if is_echo_annotated(view, current_echo_id):
            print(f'EEcho {current_echo_id} with view {view} has already been annotated. Skipping.')
            current_index += 1  # Skip to the next
            continue  # Skip to the next iteration if already annotated

        if is_echo_discarded(current_echo_id):
            print(f'The echo {current_echo_id} has been discarded. Skipping.')
            current_index += 1  # Skip to the next
            continue  # Skip if already discarded

        echo_folder = os.path.join(dataset_path, current_echo_id)
        
        if os.path.isdir(echo_folder):
            # Get all .npy files in the folder
            npy_files = [f for f in os.listdir(echo_folder) if f.endswith('.npy')]
            
            if npy_files:
                # Show the first frame (index 0)
                current_file_index = 0
                plt.figure(figsize=(10,8))
                show_frame()
                
                # Connect keyboard, click, and mouse movement events
                plt.gcf().canvas.mpl_connect('key_press_event', on_key)
                plt.gcf().canvas.mpl_connect('button_press_event', on_click)
                plt.gcf().canvas.mpl_connect('motion_notify_event', on_motion)
                
                plt.show()  # Show the plot and wait for user interaction
                current_index += 1  # Move to the next echo_id after processing this one
                break
            else:
                print(f'No .npy files found in the folder for {current_echo_id}')
        else:
            print(f'The folder for {current_echo_id} does not exist in the dataset')
        current_index += 1
# Mouse movement event to draw the interactive box
def on_motion(event):
    global first_click, drawing_box
    
    if drawing_box and first_click is not None and event.inaxes:
        plt.clf()  # Clear the image to update the box drawing
        show_frame()  # Show the current frame again
        
        # Draw the temporary box between the first click and the current cursor position
        current_x, current_y = event.xdata, event.ydata
        plt.gca().add_patch(plt.Rectangle(first_click, current_x - first_click[0], current_y - first_click[1], 
                                          fill=False, edgecolor='red', linewidth=2))
        plt.draw()

# Start the process with the first 'echo_id'
advance_to_next_echo()
