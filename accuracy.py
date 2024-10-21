import os
import numpy as np

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

# Function to check if predicted box is within a 10% error margin of the ground truth box
def within_tolerance(pred_box, gt_box, tolerance=0.1):
    pred_x_center = (pred_box[0] + pred_box[2]) / 2
    pred_y_center = (pred_box[1] + pred_box[3]) / 2
    gt_x_center = (gt_box[0] + gt_box[2]) / 2
    gt_y_center = (gt_box[1] + gt_box[3]) / 2
    
    pred_width = pred_box[2] - pred_box[0]
    pred_height = pred_box[3] - pred_box[1]
    gt_width = gt_box[2] - gt_box[0]
    gt_height = gt_box[3] - gt_box[1]

    # Avoid division by zero
    if gt_width == 0 or gt_height == 0:
        return False

    width_error = abs(pred_width - gt_width) / gt_width <= tolerance
    height_error = abs(pred_height - gt_height) / gt_height <= tolerance
    
    return width_error and height_error


# Function to evaluate predictions by comparing predicted boxes with ground truth boxes
def evaluate_predictions(predicted_folder, ground_truth_folder, iou_threshold=0.5):
    TP, FP, FN = 0, 0, 0
    total_ground_truths = 0
    total_predictions = 0

    # Loop through each label file in the predicted folder
    for pred_file in os.listdir(predicted_folder):
        pred_file_path = os.path.join(predicted_folder, pred_file)
        gt_file_path = os.path.join(ground_truth_folder, pred_file.replace('.txt', '.txt'))

        # Load predicted boxes
        pred_boxes = []
        if os.path.exists(pred_file_path):
            with open(pred_file_path, 'r') as f:
                for line in f:
                    coords = list(map(float, line.strip().split()[1:5]))  # Ignore class id
                    pred_boxes.append(coords)
                    total_predictions += 1  # Count each predicted bounding box

        # Load ground truth boxes
        gt_boxes = []
        if os.path.exists(gt_file_path):
            with open(gt_file_path, 'r') as f:
                for line in f:
                    coords = list(map(float, line.strip().split()[1:5]))  # Ignore class id
                    gt_boxes.append(coords)
                    total_ground_truths += 1  # Count each ground truth bounding box

        matched_gt = []
        
        # Check for true positives and false positives
        for pred_box in pred_boxes:
            match_found = False
            for gt_box in gt_boxes:
                if calculate_iou(pred_box, gt_box) >= iou_threshold or within_tolerance(pred_box, gt_box):
                    TP += 1
                    matched_gt.append(gt_box)
                    match_found = True
                    break
            if not match_found:
                FP += 1
        
        # Check for false negatives
        for gt_box in gt_boxes:
            if gt_box not in matched_gt:
                FN += 1

    # Calculate true negatives (TN)
    TN = max(0, total_ground_truths - TP)

    # Calculate normalized metrics
    normalized_precision, normalized_recall, normalized_f1, normalized_accuracy = calculate_normalized_metrics(
        TP, FP, FN, TN, total_ground_truths, total_predictions
    )

    print(f"Total Ground Truth Bounding Boxes: {total_ground_truths}")
    print(f"Total Predicted Bounding Boxes: {total_predictions}")
    print(f"True Positives: {TP}")
    print(f"False Positives: {FP}")
    print(f"False Negatives: {FN}")
    print(f"True Negatives: {TN}")
    print(f"Normalized Precision: {normalized_precision}")
    print(f"Normalized Recall: {normalized_recall}")
    print(f"Normalized F1 Score: {normalized_f1}")
    print(f"Normalized Accuracy: {normalized_accuracy}")

    return total_ground_truths, total_predictions, TP, FP, FN, TN


# Function to calculate normalized metrics
def calculate_normalized_metrics(TP, FP, FN, TN, total_ground_truths, total_predictions):
    # Basic metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # "Normalized" metrics (optional)
    normalized_precision = precision  # Precision is already a normalized value
    normalized_recall = recall  # Recall is already a normalized value
    normalized_f1 = f1  # F1 score is already a normalized value
    normalized_accuracy = accuracy  # Accuracy is already a normalized value

    return (normalized_precision, normalized_recall, normalized_f1, normalized_accuracy)


# Main function to run the evaluation and save metrics
def main(predicted_folder, ground_truth_folder):
    total_ground_truths, total_predictions, TP, FP, FN, TN = evaluate_predictions(predicted_folder, ground_truth_folder)

    # Calculate metrics
    normalized_precision, normalized_recall, normalized_f1, normalized_accuracy = calculate_normalized_metrics(
        TP, FP, FN, TN, total_ground_truths, total_predictions
    )

    # Prepare metrics dictionary
    metrics = {
        "Total Ground Truth Bounding Boxes": total_ground_truths,
        "Total Predicted Bounding Boxes": total_predictions,
        "True Positives": TP,
        "False Positives": FP,
        "False Negatives": FN,
        "True Negatives": TN,
        "Normalized Precision": normalized_precision,
        "Normalized Recall": normalized_recall,
        "Normalized F1 Score": normalized_f1,
        "Normalized Accuracy": normalized_accuracy,
    }

    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save metrics to a text file
    with open('evaluation_metrics.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")


# Define paths to predicted and ground truth label folders
predicted_folder = "C:/Users/Neha KB/Desktop/custom/predicted_labels/"
ground_truth_folder = "C:/Users/Neha KB/Desktop/custom/c_train/valid/labels/"

# Run the evaluation
main(predicted_folder, ground_truth_folder)
