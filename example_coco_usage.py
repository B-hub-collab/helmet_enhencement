#!/usr/bin/env python3
"""
Example script showing how to use prepare_training_data.py with COCO annotations
"""

import os
import subprocess
import sys

def run_coco_data_preparation():
    """
    Example of preparing training data from COCO annotations
    """
    
    # Your paths
    annotation_file = "/home/brinno_user/boyatry/mmdetection/dataset/PPE_detect/annotations/train.json"
    images_dir = "/home/brinno_user/boyatry/mmdetection/dataset/PPE_detect/train2017/"
    output_dir = "./helmet_training_data_from_coco"
    
    # Command to run
    cmd = [
        "python", "prepare_training_data.py",
        "--mode", "coco",
        "--annotation_file", annotation_file,
        "--images_dir", images_dir,
        "--output_dir", output_dir,
        "--worker_class_id", "0",     # Worker class ID
        "--helmet_class_id", "2",     # Helmet class ID
        "--balance_dataset",          # Balance the dataset
        "--min_samples_per_class", "50"
    ]
    
    print("Running COCO-based data preparation...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Success!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True

def run_detection_based_preparation():
    """
    Example of the original detection-based preparation (for comparison)
    """
    
    # Your detection model paths
    model_path = "/home/brinno_user/models/CHVSODASOD.pth"
    config_path = "/home/brinno_user/work_dirs/dino-4scale_r50_8xb2-24e_coco/CHVSODASOD_config.py"
    images_dir = "/home/brinno_user/boyatry/mmdetection/dataset/PPE_detect/train2017/"
    output_dir = "./helmet_training_data_from_detection"
    
    # Command to run
    cmd = [
        "python", "prepare_training_data.py",
        "--mode", "detection",
        "--model_path", model_path,
        "--config_path", config_path,
        "--images_dir", images_dir,
        "--output_dir", output_dir,
        "--use_pose",                 # Use YOLO Pose
        "--balance_dataset",
        "--min_samples_per_class", "50"
    ]
    
    print("Running detection-based data preparation...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Success!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True

def main():
    print("=== Helmet Classification Data Preparation Examples ===")
    print()
    
    choice = input("Choose preparation method:\n1. COCO annotations (recommended)\n2. Detection model\n3. Both\nEnter choice (1/2/3): ")
    
    if choice == "1":
        run_coco_data_preparation()
    elif choice == "2":
        run_detection_based_preparation()
    elif choice == "3":
        print("Running both methods...")
        print("\n--- COCO Method ---")
        success1 = run_coco_data_preparation()
        print("\n--- Detection Method ---")
        success2 = run_detection_based_preparation()
        
        if success1 and success2:
            print("\n=== Comparison ===")
            print("You can now compare the results from both methods:")
            print("- COCO method: ./helmet_training_data_from_coco/")
            print("- Detection method: ./helmet_training_data_from_detection/")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()