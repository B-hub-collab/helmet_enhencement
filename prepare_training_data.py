import os
import cv2
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm


def load_coco_annotations(annotation_file: str) -> Dict:
    """
    Load COCO format annotation file
    """
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes
    COCO format: [x, y, width, height]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to xyxy format
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def is_helmet_on_worker(worker_bbox: List[float], helmet_bbox: List[float],
                        iou_threshold: float = 0.01, spatial_threshold: float = 0.25) -> Tuple[bool, float]:
    """
    Determine if helmet is worn by worker based on spatial relationship
    Args:
        worker_bbox: [x, y, width, height] in COCO format
        helmet_bbox: [x, y, width, height] in COCO format
        iou_threshold: minimum IoU for considering helmet-worker association
        spatial_threshold: spatial relationship threshold
    Returns:
        (is_wearing_helmet, confidence_score)
    """
    # Calculate IoU
    iou = calculate_iou(worker_bbox, helmet_bbox)

    if iou < iou_threshold:
        return False, 0.0

    # Convert to center coordinates
    worker_x, worker_y, worker_w, worker_h = worker_bbox
    helmet_x, helmet_y, helmet_w, helmet_h = helmet_bbox

    worker_cx = worker_x + worker_w / 2
    worker_cy = worker_y + worker_h / 2
    helmet_cx = helmet_x + helmet_w / 2
    helmet_cy = helmet_y + helmet_h / 2

    # Check if helmet is in the upper part of worker (more flexible)
    expected_head_y = worker_y + worker_w * 0.0  # keep reference
    expected_head_y = worker_y + worker_h * 0.3  # Head should be in upper 30% of worker

    # Horizontal alignment
    horizontal_distance = abs(helmet_cx - worker_cx)
    horizontal_score = max(0, 1.0 - (horizontal_distance / (worker_w * 0.5)))

    # Vertical position (helmet should be above or in upper part of worker)
    if helmet_cy <= expected_head_y:
        vertical_score = 1.0
    else:
        vertical_distance = helmet_cy - expected_head_y
        vertical_score = max(0, 1.0 - (vertical_distance / (worker_h * 0.3)))

    # Size relationship (helmet should be reasonable size relative to worker)
    helmet_area = helmet_w * helmet_h
    worker_area = worker_w * worker_h
    size_ratio = helmet_area / worker_area
    size_score = 1.0 if 0.005 <= size_ratio <= 0.4 else 0.7  # More lenient size range

    # Combined confidence
    confidence = (horizontal_score * 0.4 + vertical_score * 0.4 + size_score * 0.2 + iou * 0.2)
    is_wearing = confidence >= spatial_threshold

    return is_wearing, confidence


def extract_head_from_worker_bbox(worker_bbox: List[float], image_shape: Tuple[int, int],
                                  min_head_size: int = 64) -> List[float]:
    """
    Extract head region from worker bounding box with minimum size enforcement
    Args:
        worker_bbox: [x, y, width, height] in COCO format
        image_shape: (height, width) of image
        min_head_size: minimum head region size in pixels
    Returns:
        head_roi: [x1, y1, x2, y2] in xyxy format
    """
    x, y, w, h = worker_bbox

    # Calculate initial head region
    head_height = h * 0.25
    head_width = w * 0.8

    center_x = x + w / 2
    head_center_y = y + h * 0.15  # Head center at 15% from top

    # Ensure minimum head size
    if head_height < min_head_size:
        head_height = min(min_head_size, h * 0.4)  # Cap at 40% of worker height

    if head_width < min_head_size:
        head_width = min(min_head_size, w * 1.2)  # Allow slight expansion beyond worker width

    head_x1 = max(0, center_x - head_width / 2)
    head_y1 = max(0, head_center_y - head_height / 2)
    head_x2 = min(image_shape[1], center_x + head_width / 2)
    head_y2 = min(image_shape[0], head_center_y + head_height / 2)

    # Final size check and adjustment
    actual_width = head_x2 - head_x1
    actual_height = head_y2 - head_y1

    if actual_width < min_head_size or actual_height < min_head_size:
        # Expand around center, respecting image boundaries
        expand_x = max(0, (min_head_size - actual_width) / 2)
        expand_y = max(0, (min_head_size - actual_height) / 2)

        head_x1 = max(0, head_x1 - expand_x)
        head_y1 = max(0, head_y1 - expand_y)
        head_x2 = min(image_shape[1], head_x2 + expand_x)
        head_y2 = min(image_shape[0], head_y2 + expand_y)

    return [head_x1, head_y1, head_x2, head_y2]


def process_coco_annotations(annotation_file: str,
                             images_dir: str,
                             output_dir: str,
                             worker_class_id: int = 0,
                             helmet_class_id: int = 2,
                             overlap_rule: bool = False) -> List[Dict]:
    """
    Process COCO annotations to extract head regions and determine helmet wearing status
    If overlap_rule is True: any IoU > 0 between worker and helmet counts as wearing.
    """
    print(f"Loading annotations from: {annotation_file}")
    coco_data = load_coco_annotations(annotation_file)

    # Create mappings
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    print(f"Found {len(annotations_by_image)} images with annotations")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    extracted_data = []

    for image_id, annotations in tqdm(annotations_by_image.items(), desc="Processing images"):
        if image_id not in image_id_to_info:
            continue

        image_info = image_id_to_info[image_id]
        image_path = os.path.join(images_dir, image_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Separate workers and helmets
        workers = [ann for ann in annotations if ann['category_id'] == worker_class_id]
        helmets = [ann for ann in annotations if ann['category_id'] == helmet_class_id]

        # Process each worker
        for worker_idx, worker_ann in enumerate(workers):
            worker_bbox = worker_ann['bbox']  # [x, y, width, height]

            # Find best matching helmet
            best_helmet = None
            best_confidence = 0.0
            helmet_wearing = False

            for helmet_ann in helmets:
                helmet_bbox = helmet_ann['bbox']

                if overlap_rule:
                    # Simple rule: any overlap counts as wearing
                    iou = calculate_iou(worker_bbox, helmet_bbox)
                    if iou > 0 and iou > best_confidence:
                        best_confidence = iou  # Use IoU as confidence (0~1)
                        best_helmet = helmet_ann
                        helmet_wearing = True
                else:
                    # Original (spatial) rule
                    wearing, confidence = is_helmet_on_worker(worker_bbox, helmet_bbox)
                    if wearing and confidence > best_confidence:
                        best_confidence = confidence
                        best_helmet = helmet_ann
                        helmet_wearing = True

            # Extract head region
            head_roi = extract_head_from_worker_bbox(worker_bbox, image.shape)

            # Extract head region from image
            x1, y1, x2, y2 = [int(coord) for coord in head_roi]
            x1, x2 = max(0, min(x1, image.shape[1])), max(0, min(x2, image.shape[1]))
            y1, y2 = max(0, min(y1, image.shape[0])), max(0, min(y2, image.shape[0]))

            if x2 <= x1 or y2 <= y1:
                continue

            head_region = image[y1:y2, x1:x2]
            if head_region.size == 0:
                continue

            # Resize if too small
            if head_region.shape[0] < 32 or head_region.shape[1] < 32:
                head_region = cv2.resize(head_region, (64, 64))

            # Determine label based on helmet matching
            if overlap_rule:
                # Overlap rule: any overlap -> wearing, ignore confidence threshold
                label = "wearing_helmet" if helmet_wearing else "no_helmet"
            else:
                # Original thresholds with uncertain band
                if helmet_wearing and best_confidence >= 0.4:
                    label = "wearing_helmet"
                elif not helmet_wearing or best_confidence < 0.15:
                    label = "no_helmet"
                else:
                    label = "uncertain"

            # Save head region
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            base_name = os.path.splitext(image_info['file_name'])[0]
            head_image_name = f"{base_name}_worker_{worker_idx}.jpg"
            head_image_path = os.path.join(label_dir, head_image_name)

            cv2.imwrite(head_image_path, head_region)

            # Store metadata
            extracted_data.append({
                "original_image": image_info['file_name'],
                "image_id": image_id,
                "worker_id": f"worker_{worker_idx}",
                "worker_annotation_id": worker_ann['id'],
                "head_image_path": head_image_path,
                "head_roi": head_roi,
                "worker_bbox": worker_bbox,
                "label": label,
                "confidence": float(best_confidence),
                "helmet_wearing": helmet_wearing,
                "matched_helmet_id": best_helmet['id'] if best_helmet else None,
                "matched_helmet_bbox": best_helmet['bbox'] if best_helmet else None,
                "head_extraction_method": "bbox_estimation",
                "analysis_details": {
                    "rule": "overlap" if overlap_rule else "spatial",
                    "spatial_confidence": float(best_confidence),
                    "total_helmets_in_image": len(helmets),
                    "worker_bbox": worker_bbox
                }
            })

    print(f"Extracted {len(extracted_data)} head regions")
    return extracted_data


def extract_head_regions_from_enhanced_results(image: np.ndarray,
                                               enhanced_results: Dict,
                                               output_dir: str,
                                               image_name: str) -> List[Dict]:
    """
    Extract head regions from enhanced detection results and save them as training images
    """
    extracted_data = []

    for worker_idx, worker in enumerate(enhanced_results["workers"]):
        # Get head region coordinates
        head_roi = worker["head_roi"]
        worker_id = worker["worker_id"]

        # Extract head region from image
        x1, y1, x2, y2 = [int(coord) for coord in head_roi]
        x1, x2 = max(0, min(x1, image.shape[1])), max(0, min(x2, image.shape[1]))
        y1, y2 = max(0, min(y1, image.shape[0])), max(0, min(y2, image.shape[0]))

        if x2 <= x1 or y2 <= y1:
            print(f"Invalid head region for {worker_id} in {image_name}")
            continue

        head_region = image[y1:y2, x1:x2]

        if head_region.size == 0:
            print(f"Empty head region for {worker_id} in {image_name}")
            continue

        # Resize to minimum size if too small
        if head_region.shape[0] < 32 or head_region.shape[1] < 32:
            head_region = cv2.resize(head_region, (64, 64))

        # Determine label based on helmet status
        helmet_status = worker["helmet_status"]
        confidence = worker["confidence"]

        # Only use high-confidence predictions for automatic labeling
        if helmet_status == "wearing_helmet" and confidence >= 0.7:
            label = "wearing_helmet"
        elif helmet_status == "no_helmet" and confidence >= 0.7:
            label = "no_helmet"
        else:
            label = "uncertain"  # These will need manual annotation

        # Create output directory for this label
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # Save head region image
        base_name = os.path.splitext(image_name)[0]
        head_image_name = f"{base_name}_{worker_id}.jpg"
        head_image_path = os.path.join(label_dir, head_image_name)

        cv2.imwrite(head_image_path, head_region)

        # Store metadata
        extracted_data.append({
            "original_image": image_name,
            "worker_id": worker_id,
            "head_image_path": head_image_path,
            "head_roi": head_roi,
            "label": label,
            "confidence": confidence,
            "helmet_status": helmet_status,
            "head_extraction_method": worker.get("head_extraction_method", "simple"),
            "analysis_details": worker["analysis_details"]
        })

    return extracted_data


def process_images_directory(images_dir: str,
                             output_dir: str,
                             model_path: str,
                             config_path: Optional[str] = None,
                             use_pose: bool = True,
                             pose_model_path: Optional[str] = None) -> List[Dict]:
    """
    Process all images in a directory and extract head regions for training
    """
    all_extracted_data = []

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
        image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No images found in {images_dir}")
        return all_extracted_data

    print(f"Found {len(image_files)} images to process")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            image_path = str(image_file)
            image_name = image_file.name

            # Run enhanced detection
            from helmet_detection_2 import enhance_detection_with_helmet_analysis, HelmetDetectionEnhancer

            enhanced_results, original_image = enhance_detection_with_helmet_analysis(
                model_path=model_path,
                image_path=image_path,
                config_path=config_path,
                modeltype="mmdet",
                pose_model_path=pose_model_path,
                use_pose=use_pose
            )

            # Extract head regions
            extracted_data = extract_head_regions_from_enhanced_results(
                original_image, enhanced_results, output_dir, image_name
            )

            all_extracted_data.extend(extracted_data)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

    return all_extracted_data


def create_annotation_interface_data(extracted_data: List[Dict],
                                     output_file: str):
    """
    Create data file for manual annotation interface
    """
    uncertain_data = [item for item in extracted_data if item["label"] == "uncertain"]

    annotation_data = {
        "total_images": len(uncertain_data),
        "images": []
    }

    for item in uncertain_data:
        # Handle both detection mode and COCO mode data structures
        predicted_status = item.get("helmet_status", "uncertain")  # detection mode
        if predicted_status == "uncertain":
            # For COCO mode, derive status from helmet_wearing field
            if "helmet_wearing" in item:
                predicted_status = "wearing_helmet" if item["helmet_wearing"] else "no_helmet"

        annotation_data["images"].append({
            "id": len(annotation_data["images"]),
            "image_path": item["head_image_path"],
            "original_image": item["original_image"],
            "worker_id": item["worker_id"],
            "predicted_status": predicted_status,
            "confidence": item["confidence"],
            "head_extraction_method": item.get("head_extraction_method", "unknown")
        })

    with open(output_file, 'w') as f:
        json.dump(annotation_data, f, indent=2)

    print(f"Created annotation data for {len(uncertain_data)} uncertain cases: {output_file}")


def generate_dataset_statistics(extracted_data: List[Dict]) -> Dict:
    """
    Generate statistics about the extracted dataset
    """
    stats = {
        "total_images": len(extracted_data),
        "labels": {},
        "confidence_distribution": {},
        "head_extraction_methods": {},
        "original_images": set()
    }

    for item in extracted_data:
        # Count labels
        label = item["label"]
        stats["labels"][label] = stats["labels"].get(label, 0) + 1

        # Count head extraction methods
        method = item["head_extraction_method"]
        stats["head_extraction_methods"][method] = stats["head_extraction_methods"].get(method, 0) + 1

        # Collect confidence distribution (binned)
        confidence = item["confidence"]
        conf_bin = f"{int(confidence * 10) / 10:.1f}-{int(confidence * 10 + 1) / 10:.1f}"
        stats["confidence_distribution"][conf_bin] = stats["confidence_distribution"].get(conf_bin, 0) + 1

        # Count unique original images
        stats["original_images"].add(item["original_image"])

    stats["unique_original_images"] = len(stats["original_images"])
    stats.pop("original_images")  # Remove set for JSON serialization

    return stats


def balance_dataset(extracted_data: List[Dict],
                    target_ratio: float = 0.5,
                    min_samples_per_class: int = 50) -> List[Dict]:
    """
    Balance the dataset by sampling from classes
    """
    # Separate by label
    wearing_helmet = [item for item in extracted_data if item["label"] == "wearing_helmet"]
    no_helmet = [item for item in extracted_data if item["label"] == "no_helmet"]

    print(f"Original distribution: wearing_helmet={len(wearing_helmet)}, no_helmet={len(no_helmet)}")

    # Check if we have enough samples
    if len(wearing_helmet) < min_samples_per_class or len(no_helmet) < min_samples_per_class:
        print(f"Warning: Not enough samples in some classes (min required: {min_samples_per_class})")
        return extracted_data

    # Balance the dataset
    if len(wearing_helmet) > len(no_helmet):
        # More wearing_helmet samples, downsample them
        target_wearing_helmet = min(len(wearing_helmet), int(len(no_helmet) / (1 - target_ratio) * target_ratio))
        np.random.shuffle(wearing_helmet)
        wearing_helmet = wearing_helmet[:target_wearing_helmet]
    else:
        # More no_helmet samples, downsample them
        target_no_helmet = min(len(no_helmet), int(len(wearing_helmet) / target_ratio * (1 - target_ratio)))
        np.random.shuffle(no_helmet)
        no_helmet = no_helmet[:target_no_helmet]

    balanced_data = wearing_helmet + no_helmet
    np.random.shuffle(balanced_data)

    print(f"Balanced distribution: wearing_helmet={len(wearing_helmet)}, no_helmet={len(no_helmet)}")

    return balanced_data


def main():
    parser = argparse.ArgumentParser(description='Prepare Training Data for Helmet Classification')
    parser.add_argument('--mode', choices=['detection', 'coco'], default='detection',
                        help='Processing mode: detection (run detection model) or coco (use ground truth annotations)')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing images to process')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted head regions')

    # Detection mode arguments
    parser.add_argument('--model_path', type=str,
                        help='Path to detection model (.pth file) [required for detection mode]')
    parser.add_argument('--config_path', type=str,
                        help='Path to detection model config (.py file)')
    parser.add_argument('--pose_model_path', type=str,
                        help='Path to YOLO Pose model')
    parser.add_argument('--use_pose', action='store_true', default=True,
                        help='Use YOLO Pose for head region extraction')

    # COCO mode arguments
    parser.add_argument('--annotation_file', type=str,
                        help='Path to COCO format annotation file [required for coco mode]')
    parser.add_argument('--worker_class_id', type=int, default=0,
                        help='Class ID for workers in COCO annotations')
    parser.add_argument('--helmet_class_id', type=int, default=2,
                        help='Class ID for helmets in COCO annotations')
    parser.add_argument('--overlap_rule', action='store_true',
                        help='If set, any overlap (IoU>0) between helmet and worker counts as wearing')

    # Common arguments
    parser.add_argument('--balance_dataset', action='store_true',
                        help='Balance the dataset after extraction')
    parser.add_argument('--min_samples_per_class', type=int, default=50,
                        help='Minimum samples per class for balancing')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'detection' and not args.model_path:
        print("Error: --model_path is required for detection mode")
        return

    if args.mode == 'coco' and not args.annotation_file:
        print("Error: --annotation_file is required for coco mode")
        return

    print("Starting data preparation...")
    print(f"Mode: {args.mode}")
    print(f"Images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")

    if args.mode == 'detection':
        print(f"Detection model: {args.model_path}")
        print(f"Use YOLO Pose: {args.use_pose}")

        # Import here to avoid dependency issues in coco mode
        from helmet_detection_2 import enhance_detection_with_helmet_analysis, HelmetDetectionEnhancer

        # Process images using detection
        extracted_data = process_images_directory(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            config_path=args.config_path,
            use_pose=args.use_pose,
            pose_model_path=args.pose_model_path
        )

    elif args.mode == 'coco':
        print(f"Annotation file: {args.annotation_file}")
        print(f"Worker class ID: {args.worker_class_id}")
        print(f"Helmet class ID: {args.helmet_class_id}")
        print(f"Overlap rule: {args.overlap_rule}")

        # Process images using COCO annotations
        extracted_data = process_coco_annotations(
            annotation_file=args.annotation_file,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            worker_class_id=args.worker_class_id,
            helmet_class_id=args.helmet_class_id,
            overlap_rule=args.overlap_rule
        )

    if not extracted_data:
        print("No data extracted. Exiting.")
        return

    # Generate statistics
    print("\nGenerating dataset statistics...")
    stats = generate_dataset_statistics(extracted_data)
    print(f"Total extracted images: {stats['total_images']}")
    print(f"Label distribution: {stats['labels']}")
    print(f"Head extraction methods: {stats['head_extraction_methods']}")
    print(f"From {stats['unique_original_images']} original images")

    # Save statistics
    stats_file = os.path.join(args.output_dir, "dataset_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_file}")

    # Save metadata
    metadata_file = os.path.join(args.output_dir, "extraction_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(extracted_data, f, indent=2)
    print(f"Extraction metadata saved to: {metadata_file}")

    # Create annotation interface data for uncertain cases
    annotation_file = os.path.join(args.output_dir, "uncertain_annotations.json")
    create_annotation_interface_data(extracted_data, annotation_file)

    # Balance dataset if requested
    if args.balance_dataset:
        print("\nBalancing dataset...")
        # Filter out uncertain labels for training
        training_data = [item for item in extracted_data if item["label"] in ["wearing_helmet", "no_helmet"]]

        if len(training_data) > 0:
            balanced_data = balance_dataset(training_data, min_samples_per_class=args.min_samples_per_class)

            # Save balanced metadata
            balanced_metadata_file = os.path.join(args.output_dir, "balanced_metadata.json")
            with open(balanced_metadata_file, 'w') as f:
                json.dump(balanced_data, f, indent=2)
            print(f"Balanced metadata saved to: {balanced_metadata_file}")
        else:
            print("No training data available for balancing")

    print("\nData preparation completed!")
    print(f"Check the output directory: {args.output_dir}")
    print("Directory structure:")
    print("  ├── wearing_helmet/     # Head regions with helmets")
    print("  ├── no_helmet/         # Head regions without helmets")
    print("  ├── uncertain/         # Uncertain cases (need manual annotation)")
    print("  ├── dataset_statistics.json")
    print("  ├── extraction_metadata.json")
    print("  └── uncertain_annotations.json")


if __name__ == "__main__":
    main()
