import os
import cv2
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

WORKER_CLASS_ID = 0
HARDHAT_CLASS_ID = 2


class HelmetDetectionEnhancer:
    def __init__(self, pose_model_path: Optional[str] = None, classifier_model_path: Optional[str] = None):
        self.helmet_colors_hsv = {
            "yellow": ([20, 100, 100], [30, 255, 255]),
            "white": ([0, 0, 200], [180, 30, 255]),
            "red": ([0, 100, 100], [10, 255, 255]),
            "blue": ([100, 100, 100], [130, 255, 255]),
            "orange": ([10, 100, 100], [20, 255, 255]),
            "green": ([40, 100, 100], [80, 255, 255]),
        }
        
        # Initialize YOLO Pose model
        self.pose_model = None
        if pose_model_path and YOLO is not None:
            try:
                self.pose_model = YOLO(pose_model_path)
                print(f"YOLO Pose model loaded from: {pose_model_path}")
            except Exception as e:
                print(f"Failed to load YOLO Pose model: {e}")
                import traceback
                traceback.print_exc()
                self.pose_model = None
        elif pose_model_path and YOLO is None:
            print("YOLO Pose model path provided but ultralytics not available")
        
        # Initialize classifier model
        self.classifier_model = None
        self.classifier_transform = None
        if classifier_model_path:
            try:
                # Import the HelmetClassifier class
                from helmet_classifier import HelmetClassifier
                
                # Load checkpoint
                checkpoint = torch.load(classifier_model_path, map_location='cpu')
                
                # Get model configuration
                model_config = checkpoint.get('model_config', {})
                backbone = model_config.get('backbone', 'convnext_tiny')
                num_classes = model_config.get('num_classes', 2)
                dropout_rate = model_config.get('dropout_rate', 0.2)
                
                # Create model
                self.classifier_model = HelmetClassifier(
                    backbone=backbone,
                    num_classes=num_classes,
                    pretrained=False,
                    dropout_rate=dropout_rate
                )
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.classifier_model.load_state_dict(checkpoint)
                
                self.classifier_model.eval()
                
                # Standard transform for helmet classification
                self.classifier_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print(f"Classifier model loaded from: {classifier_model_path}")
                print(f"Model config: {model_config}")
                
            except Exception as e:
                print(f"Failed to load classifier model: {e}")
                import traceback
                traceback.print_exc()
                self.classifier_model = None

    def extract_head_region_simple(self, worker_bbox: List[float], image_shape: Tuple[int, int]) -> List[float]:
        x1, y1, x2, y2 = worker_bbox
        worker_width = x2 - x1
        worker_height = y2 - y1

        # Use full worker width for helmet detection
        head_x1 = max(0, x1)
        head_x2 = min(image_shape[1], x2)
        
        # Start from slightly below estimated head position (around neck area)
        estimated_head_bottom = y1 + worker_height * 0.25  # Bottom of head area
        
        # Extend from worker top to slightly below head (to capture helmet fully)
        head_y1 = max(0, y1)  # Worker top (includes helmet area)
        head_y2 = min(image_shape[0], estimated_head_bottom)  # Down to neck area
        
        return [head_x1, head_y1, head_x2, head_y2]
    
    def extract_head_region_with_pose(self, image: np.ndarray, worker_bbox: List[float]) -> Tuple[List[float], str]:
        """Extract head region using YOLO Pose model for more accurate head detection"""
        if self.pose_model is None:
            return self.extract_head_region_simple(worker_bbox, image.shape), "simple"
        
        try:
            # Crop worker region for pose estimation
            x1, y1, x2, y2 = [int(c) for c in worker_bbox]
            worker_crop = image[y1:y2, x1:x2]
            
            if worker_crop.size == 0:
                return self.extract_head_region_simple(worker_bbox, image.shape), "simple_fallback"
            
            # Run YOLO Pose on worker crop
            results = self.pose_model(worker_crop, verbose=False)
            
            if len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy[0].cpu().numpy()  # Get first person's keypoints and move to CPU
                
                # Head keypoints: nose=0, left_eye=1, right_eye=2, left_ear=3, right_ear=4
                head_keypoint_indices = [0, 1, 2, 3, 4]
                valid_head_points = []
                
                for idx in head_keypoint_indices:
                    if idx < len(keypoints):
                        kp = keypoints[idx]
                        if kp[0] > 0 and kp[1] > 0:  # Valid keypoint
                            # Convert back to original image coordinates
                            orig_x = x1 + kp[0]
                            orig_y = y1 + kp[1]
                            valid_head_points.append([orig_x, orig_y])
                
                if len(valid_head_points) >= 2:
                    # Get worker bbox boundaries
                    wx1, wy1, wx2, wy2 = [int(c) for c in worker_bbox]
                    
                    # Use full worker width for consistency
                    head_x1 = max(0, wx1)
                    head_x2 = min(image.shape[1], wx2)
                    
                    # Calculate head keypoints bounds
                    valid_head_points = np.array(valid_head_points)
                    min_x, min_y = valid_head_points.min(axis=0)
                    max_x, max_y = valid_head_points.max(axis=0)
                    
                    # Use worker top and extend to below the lowest head keypoint
                    head_y1 = max(0, wy1)  # Worker top (includes helmet area)  
                    head_y2 = min(image.shape[0], max_y + 15)  # Slightly below lowest keypoint
                    
                    return [head_x1, head_y1, head_x2, head_y2], "pose"
        
        except Exception as e:
            print(f"YOLO Pose extraction failed: {e}")
        
        # Fallback to simple method
        return self.extract_head_region_simple(worker_bbox, image.shape), "simple_fallback"

    def helmet_worker_spatial_matching(
        self,
        worker_bbox: List[float],
        hardhat_detections: List[Dict],
        threshold: float = 0.3,
    ) -> Tuple[Optional[Dict], float]:
        if not hardhat_detections:
            return None, 0.0

        wx1, wy1, wx2, wy2 = worker_bbox
        wcx = (wx1 + wx2) / 2
        w_width = wx2 - wx1
        w_height = wy2 - wy1
        expected_head_y = wy1 + w_height * 0.2

        best_match, best_score = None, 0.0
        for hh in hardhat_detections:
            hx1, hy1, hx2, hy2 = hh["bbox"]
            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2
            hconf = hh["score"]

            score = 0.0
            hdist = abs(hcx - wcx)
            if hdist < w_width * 0.4:
                score += 0.4 * (1.0 - (hdist / (w_width * 0.4)))

            if hcy <= expected_head_y:
                vdist = abs(hcy - expected_head_y)
                if vdist < w_height * 0.3:
                    score += 0.4 * (1.0 - (vdist / (w_height * 0.3)))

            hardhat_area = (hx2 - hx1) * (hy2 - hy1)
            worker_area = w_width * w_height
            ratio = hardhat_area / max(worker_area, 1e-6)
            if 0.01 <= ratio <= 0.2:
                score += 0.1

            score += 0.1 * float(hconf)

            if score > best_score and score > threshold:
                best_score = score
                best_match = hh

        return best_match, best_score

    def analyze_head_region_color(self, image: np.ndarray, head_roi: List[float]) -> Dict:
        x1, y1, x2, y2 = [int(c) for c in head_roi]
        H, W = image.shape[:2]
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        if x2 <= x1 or y2 <= y1:
            return {"has_helmet_color": False, "confidence": 0.0, "dominant_color": None, "color_ratio": 0.0}

        head_region = image[y1:y2, x1:x2]
        if head_region.size == 0:
            return {"has_helmet_color": False, "confidence": 0.0, "dominant_color": None, "color_ratio": 0.0}

        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        total = head_region.shape[0] * head_region.shape[1]
        helmet_pixel_count, dominant_color, max_pixels = 0, None, 0

        for cname, (lower, upper) in self.helmet_colors_hsv.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            cnt = int(np.sum(mask > 0))
            helmet_pixel_count += cnt
            if cnt > max_pixels:
                max_pixels, dominant_color = cnt, cname

        ratio = helmet_pixel_count / total if total > 0 else 0.0
        return {
            "has_helmet_color": ratio > 0.15,
            "confidence": min(ratio * 2, 1.0),
            "dominant_color": dominant_color,
            "color_ratio": ratio,
        }

    def analyze_head_region_negative(self, image: np.ndarray, head_roi: List[float]) -> Dict:
        x1, y1, x2, y2 = [int(c) for c in head_roi]
        H, W = image.shape[:2]
        x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
        y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
        if x2 <= x1 or y2 <= y1:
            return {"skin_ratio": 0.0}

        head = image[y1:y2, x1:x2]
        if head.size == 0:
            return {"skin_ratio": 0.0}

        hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
        lower1, upper1 = np.array([0, 48, 80]),  np.array([20, 255, 255])
        lower2, upper2 = np.array([170, 48, 80]), np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        skin_mask = cv2.bitwise_or(mask1, mask2)

        total = head.shape[0] * head.shape[1]
        skin_ratio = float(np.sum(skin_mask > 0)) / total if total > 0 else 0.0
        return {"skin_ratio": skin_ratio}
    
    def classify_head_with_deep_learning(self, image: np.ndarray, head_roi: List[float]) -> Optional[Dict]:
        """Use trained classifier model to classify head region"""
        if self.classifier_model is None or self.classifier_transform is None:
            return None
        
        try:
            # Extract head region
            x1, y1, x2, y2 = [int(c) for c in head_roi]
            head_crop = image[y1:y2, x1:x2]
            
            if head_crop.size == 0:
                return None
            
            # Convert BGR to RGB for PIL
            head_crop_rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_tensor = self.classifier_transform(head_crop_rgb).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.classifier_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # Assuming binary classification: 0=no_helmet, 1=wearing_helmet
                class_names = ["no_helmet", "wearing_helmet"]
                predicted_class = class_names[predicted.item()]
                confidence_score = confidence.item()
                
                # Map to expected format for fusion logic
                mapped_class = "helmet" if predicted_class == "wearing_helmet" else "no_helmet"
                
                return {
                    "class_name": mapped_class,
                    "confidence": confidence_score,
                    "raw_probabilities": probabilities[0].cpu().numpy().tolist(),
                    "original_class": predicted_class
                }
        
        except Exception as e:
            print(f"Deep learning classification failed: {e}")
            return None

    def enhance_helmet_detection(self, image: np.ndarray, detection_results: Dict, use_pose: bool = False, use_classifier: bool = False) -> Dict:
        enhanced = {
            "workers": [],
            "original_detections": detection_results,
            "enhancement_summary": {
                "total_workers": 0,
                "workers_with_helmet": 0,
                "workers_without_helmet": 0,
                "uncertain_cases": 0,
                "pose_detection_used": use_pose,
                "classifier_used": use_classifier,
            },
        }

        workers = [d for d in detection_results.get("object_prediction_list", []) if d.get("category_id") == WORKER_CLASS_ID]
        hardhats = [d for d in detection_results.get("object_prediction_list", []) if d.get("category_id") == HARDHAT_CLASS_ID]

        for i, worker in enumerate(workers):
            worker_bbox = worker["bbox"]
            
            # Choose head extraction method based on availability
            if use_pose and self.pose_model is not None:
                head_roi, extraction_method = self.extract_head_region_with_pose(image, worker_bbox)
                print(f"Worker {i}: Using pose extraction, result: {extraction_method}")
            else:
                head_roi = self.extract_head_region_simple(worker_bbox, image.shape)
                extraction_method = "simple"
                if use_pose:
                    print(f"Worker {i}: Pose requested but pose_model is None")

            # Run all analysis methods
            matched_hardhat, spatial_score = self.helmet_worker_spatial_matching(worker_bbox, hardhats, threshold=0.3)
            color_analysis = self.analyze_head_region_color(image, head_roi)
            neg_analysis = self.analyze_head_region_negative(image, head_roi)
            
            # Run deep learning classification if enabled
            dl_classification = None
            if use_classifier and self.classifier_model is not None:
                dl_classification = self.classify_head_with_deep_learning(image, head_roi)
                if dl_classification:
                    print(f"Worker {i}: DL Classification: {dl_classification['class_name']} ({dl_classification['confidence']:.3f})")
                else:
                    print(f"Worker {i}: DL Classification failed")
            elif use_classifier:
                print(f"Worker {i}: Classifier requested but classifier_model is None")

            status, conf = self._fuse_analysis_results(matched_hardhat, spatial_score, color_analysis, neg_analysis, dl_classification)

            worker_result = {
                "worker_id": f"worker_{i}",
                "worker_bbox": worker_bbox,
                "head_roi": head_roi,
                "helmet_status": status,
                "confidence": conf,
                "head_extraction_method": extraction_method,
                "analysis_details": {
                    "spatial_matching": {
                        "matched": matched_hardhat is not None,
                        "score": spatial_score,
                        "matched_helmet_bbox": matched_hardhat["bbox"] if matched_hardhat else None,
                    },
                    "color_analysis": color_analysis,
                    "negative_evidence": neg_analysis,
                    "deep_learning_classification": dl_classification,
                },
            }
            enhanced["workers"].append(worker_result)

            if status == "wearing_helmet":
                enhanced["enhancement_summary"]["workers_with_helmet"] += 1
            elif status == "no_helmet":
                enhanced["enhancement_summary"]["workers_without_helmet"] += 1
            else:
                enhanced["enhancement_summary"]["uncertain_cases"] += 1

        enhanced["enhancement_summary"]["total_workers"] = len(workers)
        return enhanced

    def _fuse_analysis_results(
        self,
        matched_helmet: Optional[Dict],
        spatial_score: float,
        color_analysis: Dict,
        neg_analysis: Dict,
        dl_classification: Optional[Dict] = None
    ) -> Tuple[str, float]:
        has_color = bool(color_analysis.get("has_helmet_color", False))
        color_ratio = float(color_analysis.get("color_ratio", 0.0))
        color_conf = float(color_analysis.get("confidence", 0.0))
        skin_ratio = float(neg_analysis.get("skin_ratio", 0.0))

        # Deep learning classification results
        dl_prediction = None
        dl_confidence = 0.0
        if dl_classification:
            dl_prediction = dl_classification.get("class_name")
            dl_confidence = float(dl_classification.get("confidence", 0.0))

        # Priority 1: High confidence deep learning prediction
        if dl_classification and dl_confidence >= 0.8:
            if dl_prediction == "helmet":
                return "wearing_helmet", dl_confidence
            elif dl_prediction == "no_helmet":
                return "no_helmet", dl_confidence

        # Priority 2: Strong spatial matching with supporting evidence
        if matched_helmet and spatial_score >= 0.5:
            # Enhance confidence with DL if available
            if dl_classification and dl_prediction == "helmet":
                combined_conf = min(0.95, (spatial_score + dl_confidence) * 0.6)
                return "wearing_helmet", combined_conf
            return "wearing_helmet", max(0.7, (spatial_score + color_conf) * 0.5)

        # Priority 3: Medium confidence deep learning + supporting evidence
        if dl_classification and dl_confidence >= 0.6:
            if dl_prediction == "helmet" and (has_color or spatial_score >= 0.3):
                return "wearing_helmet", max(0.65, dl_confidence * 0.9)
            elif dl_prediction == "no_helmet" and (skin_ratio >= 0.15 or spatial_score < 0.2):
                return "no_helmet", max(0.65, dl_confidence * 0.9)

        # Priority 4: Traditional color analysis
        if has_color and color_ratio >= 0.25:
            # Check if DL disagrees strongly
            if dl_classification and dl_prediction == "no_helmet" and dl_confidence >= 0.6:
                return "uncertain", 0.5
            return "wearing_helmet", max(0.6, color_conf)

        # Priority 5: Strong negative evidence
        if skin_ratio >= 0.25 and color_ratio < 0.08:
            # Check if DL disagrees strongly
            if dl_classification and dl_prediction == "helmet" and dl_confidence >= 0.6:
                return "uncertain", 0.5
            c = min(1.0, 0.5 + 0.5 * (skin_ratio - color_ratio))
            return "no_helmet", c

        # Priority 6: Weak positive evidence
        if has_color and color_ratio >= 0.15:
            return "wearing_helmet", max(0.5, color_conf)

        # Priority 7: Low confidence deep learning
        if dl_classification and dl_confidence >= 0.4:
            if dl_prediction == "helmet":
                return "wearing_helmet", dl_confidence
            elif dl_prediction == "no_helmet":
                return "no_helmet", dl_confidence

        return "uncertain", 0.5


def run_sahi_detection(
    model_path: str,
    config_path: Optional[str],
    image: np.ndarray,
    modeltype: str = "mmdet",
    conf_thres: float = 0.35,
    device: Optional[str] = None,
) -> Dict:
    if device is None:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=modeltype,
        model_path=model_path,
        config_path=config_path,
        confidence_threshold=conf_thres,
        device=device,
    )

    ip = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        postprocess_match_metric="IOU",
    )

    object_prediction_list = []
    for op in ip.object_prediction_list:
        try:
            cid = int(op.category.id)
        except Exception:
            cid = None
        try:
            cname = str(op.category.name)
        except Exception:
            cname = str(op.category)

        x1, y1, x2, y2 = op.bbox.to_xyxy()
        try:
            score = float(op.score.value)
        except Exception:
            score = float(op.score)

        object_prediction_list.append({
            "category_id": cid,
            "category_name": cname,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": score,
        })

    return {"object_prediction_list": object_prediction_list}


def visualize_enhancement(image: np.ndarray, results: Dict, save_path: Optional[str] = None) -> np.ndarray:
    vis = image.copy()

    colors = {
        "wearing_hardhat": (0, 255, 0),
        "no_hardhat": (0, 0, 255),
        "uncertain": (0, 255, 255),
    }

    status_display_map = {
        "wearing_helmet": "wearing_hardhat",
        "no_helmet": "no_hardhat",
        "uncertain": "uncertain",
    }

    for worker in results["workers"]:
        x1, y1, x2, y2 = [int(c) for c in worker["worker_bbox"]]
        hx1, hy1, hx2, hy2 = [int(c) for c in worker["head_roi"]]

        internal_status = worker["helmet_status"]
        display_status = status_display_map.get(internal_status, "uncertain")
        conf = worker["confidence"]
        color = colors.get(display_status, (128, 128, 128))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), (255, 255, 0), 1)
        cv2.putText(
            vis,
            f"{display_status}: {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        sm = worker["analysis_details"]["spatial_matching"]
        if sm["matched"] and sm["matched_helmet_bbox"]:
            mhx1, mhy1, mhx2, mhy2 = [int(c) for c in sm["matched_helmet_bbox"]]
            cv2.rectangle(vis, (mhx1, mhy1), (mhx2, mhy2), (255, 0, 255), 2)

    summary = results["enhancement_summary"]
    lines = [
        f"Total Workers: {summary['total_workers']}",
        f"With Hardhat: {summary['workers_with_helmet']}",
        f"No Hardhat: {summary['workers_without_helmet']}",
        f"Uncertain: {summary['uncertain_cases']}",
    ]
    y0 = 28
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (10, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"Enhanced results image saved to: {save_path}")
    return vis


def enhance_detection_with_helmet_analysis(
    model_path: str,
    image_path: str,
    config_path: Optional[str] = None,
    modeltype: str = "mmdet",
    pose_model_path: Optional[str] = None,
    use_pose: bool = False,
    classifier_model_path: Optional[str] = None,
    use_classifier: bool = False,
) -> Tuple[Dict, np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cannot read {image_path}")

    detections = run_sahi_detection(
        model_path=model_path,
        config_path=config_path,
        image=image,
        modeltype=modeltype,
        conf_thres=0.35,
        device=None,
    )

    enhancer = HelmetDetectionEnhancer(pose_model_path=pose_model_path, classifier_model_path=classifier_model_path)
    enhanced_results = enhancer.enhance_helmet_detection(image, detections, use_pose=use_pose, use_classifier=use_classifier)
    return enhanced_results, image


if __name__ == "__main__":
    model_path = "/home/brinno_user/models/CHVSODASOD.pth"
    mmdet_config_path = "/home/brinno_user/work_dirs/dino-4scale_r50_8xb2-24e_coco/CHVSODASOD_config.py"
    test_image_path = "/home/brinno_user/test_renew/images/-_frame_19700101_080221_000047_jpg.rf.3d08586c50ed68844a73c518e542a212.jpg"
    out_image = "hardhat_enhanced_only.jpg"

    try:
        enhanced_results, original_image = enhance_detection_with_helmet_analysis(
            model_path=model_path,
            image_path=test_image_path,
            config_path=mmdet_config_path,
            modeltype="mmdet",
        )

        print("\n=== Hardhat Detection Enhancement Results ===")
        for worker in enhanced_results["workers"]:
            print(f"Worker ID: {worker['worker_id']}")
            print(f"Status: {worker['helmet_status']}")
            print(f"Confidence: {worker['confidence']:.3f}")
            print(f"Spatial Matching: {worker['analysis_details']['spatial_matching']}")
            print(f"Color Analysis: {worker['analysis_details']['color_analysis']}")
            print(f"Negative Evidence: {worker['analysis_details']['negative_evidence']}")
            print("-" * 50)

        print(f"\nSummary: {enhanced_results['enhancement_summary']}")
        visualize_enhancement(original_image, enhanced_results, save_path=out_image)

        with open("hardhat_enhanced_results.json", "w", encoding="utf-8") as f:
            json.dump(enhanced_results, f, ensure_ascii=False, indent=2)
        print("Results JSON saved to: hardhat_enhanced_results.json")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
