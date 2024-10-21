import os
import cv2
from ultralytics import YOLO

# Function to convert bounding box format to YOLO format (x_center, y_center, width, height)
def convert_to_yolo_format(box, img_width, img_height):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [x_center, y_center, width, height]

# Function to process validation images and save predictions
def run_predictions_and_save_labels(valid_folder, labels_folder, model):
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)

    # Process each image in the validation folder
    for img_file in os.listdir(valid_folder):
        img_path = os.path.join(valid_folder, img_file)
        img = cv2.imread(img_path)

        # Predict using the YOLO model
        results = model(img)

        # Extract predicted bounding boxes
        pred_boxes = []
        for result in results:
            for box in result.boxes:
                box_xyxy = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                pred_boxes.append(box_xyxy)

        # Save the predictions to a .txt file in YOLO format
        label_file_name = img_file.replace('.jpg', '.txt')  # Adjust according to your image format
        label_file_path = os.path.join(labels_folder, label_file_name)

        img_height, img_width, _ = img.shape
        with open(label_file_path, 'w') as label_file:
            for box in pred_boxes:
                yolo_box = convert_to_yolo_format(box, img_width, img_height)
                # Write to the label file; assuming class 0 (adjust if needed)
                label_file.write(f"0 {' '.join(map(str, yolo_box))}\n")

    print(f"Predictions saved in {labels_folder}.")

# Define paths
valid_folder = "C:/Users/Neha KB/Desktop/custom/c_train/valid/images"
labels_folder = "C:/Users/Neha KB/Desktop/custom/predicted_labels/"
model = YOLO('C:/Users/Neha KB/Desktop/custom/random_search_trial_2/random_search_trial_0/weights/best.pt')

# Run the predictions and save labels
run_predictions_and_save_labels(valid_folder, labels_folder, model)
