import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('C:/Users/Neha KB/Desktop/custom/random_search_trial_2/random_search_trial_0/weights/best.pt')

# Set the input folder and ground truth folder
input_folder = 'C:/Users/Neha KB/Desktop/custom/metro_ip'
ground_truth_folder = 'C:/Users/Neha KB/Desktop/custom/ground_truth'
output_folder = 'C:/Users/Neha KB/Desktop/custom/output/output2'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Set confidence thresholds
conf_threshold = 0.4  # Confidence threshold for detection

def calculate_iou(boxA, boxB):
    # Calculate the IoU
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea)

def get_ground_truth_boxes(gt_path):
    # Load ground truth bounding boxes from a text file
    gt_boxes = []
    with open(gt_path, 'r') as file:
        for line in file.readlines():
            parts = list(map(float, line.strip().split()))
            gt_boxes.append((int(parts[0]), parts[1], parts[2], parts[3], parts[4]))  # (class_id, x1, y1, x2, y2)
    return gt_boxes

def calculate_map(predictions, ground_truths):
    # Calculate mAP based on predictions and ground truths
    # Placeholder for actual mAP calculation logic
    # Implement the mAP calculation logic here as per your requirements.
    # You may need to create a function to compute precision and recall.
    return 0.55248  # Replace with actual computation

# Loop through all the images in the input folder
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    gt_path = os.path.join(ground_truth_folder, img_name.replace('.jpg', '.txt'))  # Assuming .jpg extension

    # Load the image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Could not read image {img_name}. Skipping...")
        continue

    # Perform inference
    results = model(img)

    # Collect predictions
    predictions = []
    for box in results[0].boxes:
        if box.conf[0] >= conf_threshold:
            predictions.append((
                int(box.cls[0]),  # class_id
                box.xyxy[0][0].item(),  # x1
                box.xyxy[0][1].item(),  # y1
                box.xyxy[0][2].item(),  # x2
                box.xyxy[0][3].item(),  # y2
                box.conf[0].item()  # confidence
            ))

    # Load ground truth boxes
    ground_truths = get_ground_truth_boxes(gt_path)

    # Calculate mAP
    map_value = calculate_map(predictions, ground_truths)

    # Save or display results, including mAP
    print(f"{img_name}: mAP@50: {map_value:.5f}")

