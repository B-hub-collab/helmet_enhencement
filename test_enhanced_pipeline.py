#!/usr/bin/env python3
"""
Test script for the enhanced helmet detection pipeline
Tests the complete workflow: Detection + YOLO Pose + Deep Learning Classification
"""

import os
import sys
import cv2
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from helmet_detection_2 import enhance_detection_with_helmet_analysis, visualize_enhancement


def test_single_image(image_path: str,
                     detection_model_path: str,
                     detection_config_path: Optional[str] = None,
                     pose_model_path: Optional[str] = None,
                     classifier_model_path: Optional[str] = None,
                     output_dir: str = "./test_results") -> Dict:
    """
    Test the enhanced pipeline on a single image
    """
    print(f"Testing image: {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image name for output files
    image_name = Path(image_path).stem
    
    try:
        # Run enhanced detection with all features
        enhanced_results, original_image = enhance_detection_with_helmet_analysis(
            model_path=detection_model_path,
            image_path=image_path,
            config_path=detection_config_path,
            modeltype="mmdet",
            pose_model_path=pose_model_path,
            use_pose=pose_model_path is not None,
            classifier_model_path=classifier_model_path,
            use_classifier=classifier_model_path is not None
        )
        
        # Save results
        results_file = os.path.join(output_dir, f"{image_name}_results.json")
        with open(results_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
        
        # Generate visualization
        output_image_path = os.path.join(output_dir, f"{image_name}_enhanced.jpg")
        visualize_enhancement(original_image, enhanced_results, save_path=output_image_path)
        
        # Print summary
        summary = enhanced_results["enhancement_summary"]
        print(f"Results for {image_name}:")
        print(f"  Total workers: {summary['total_workers']}")
        print(f"  With helmet: {summary['workers_with_helmet']}")
        print(f"  Without helmet: {summary['workers_without_helmet']}")
        print(f"  Uncertain: {summary['uncertain_cases']}")
        print(f"  YOLO Pose used: {summary['pose_detection_used']}")
        print(f"  Classifier used: {summary['classifier_used']}")
        
        # Detailed worker analysis
        print(f"\\nDetailed analysis:")
        for worker in enhanced_results["workers"]:
            worker_id = worker["worker_id"]
            status = worker["helmet_status"]
            confidence = worker["confidence"]
            extraction_method = worker["head_extraction_method"]
            
            print(f"  {worker_id}:")
            print(f"    Status: {status} (confidence: {confidence:.3f})")
            print(f"    Head extraction: {extraction_method}")
            
            # Analysis details
            details = worker["analysis_details"]
            if details["spatial_matching"]["matched"]:
                print(f"    Spatial match: {details['spatial_matching']['score']:.3f}")
            
            if details["color_analysis"]["has_helmet_color"]:
                color = details["color_analysis"]["dominant_color"]
                ratio = details["color_analysis"]["color_ratio"]
                print(f"    Color detected: {color} (ratio: {ratio:.3f})")
            
            if details["deep_learning_classification"]:
                dl_conf = details["deep_learning_classification"]["confidence"]
                dl_class = details["deep_learning_classification"]["class_name"]
                print(f"    DL Classification: {dl_class} (confidence: {dl_conf:.3f})")
        
        print(f"\\nResults saved to: {results_file}")
        print(f"Visualization saved to: {output_image_path}")
        
        return enhanced_results
        
    except Exception as e:
        print(f"Error testing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def compare_methods(image_path: str,
                   detection_model_path: str,
                   detection_config_path: Optional[str] = None,
                   pose_model_path: Optional[str] = None,
                   classifier_model_path: Optional[str] = None,
                   output_dir: str = "./comparison_results") -> Dict:
    """
    Compare different method combinations on the same image
    """
    print(f"Comparing methods on: {image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(image_path).stem
    
    methods = {
        "baseline": {"use_pose": False, "use_classifier": False},
        "with_pose": {"use_pose": True, "use_classifier": False},
        "with_classifier": {"use_pose": False, "use_classifier": True},
        "full_pipeline": {"use_pose": True, "use_classifier": True}
    }
    
    results = {}
    
    for method_name, config in methods.items():
        print(f"\\nTesting {method_name}...")
        
        try:
            enhanced_results, original_image = enhance_detection_with_helmet_analysis(
                model_path=detection_model_path,
                image_path=image_path,
                config_path=detection_config_path,
                modeltype="mmdet",
                pose_model_path=pose_model_path if config["use_pose"] else None,
                use_pose=config["use_pose"],
                classifier_model_path=classifier_model_path if config["use_classifier"] else None,
                use_classifier=config["use_classifier"]
            )
            
            # Save method results
            method_file = os.path.join(output_dir, f"{image_name}_{method_name}.json")
            with open(method_file, 'w') as f:
                json.dump(enhanced_results, f, indent=2, ensure_ascii=False)
            
            # Generate visualization
            vis_file = os.path.join(output_dir, f"{image_name}_{method_name}.jpg")
            visualize_enhancement(original_image, enhanced_results, save_path=vis_file)
            
            results[method_name] = enhanced_results
            
            # Print quick summary
            summary = enhanced_results["enhancement_summary"]
            print(f"  Workers: {summary['total_workers']}, "
                  f"With helmet: {summary['workers_with_helmet']}, "
                  f"Without: {summary['workers_without_helmet']}, "
                  f"Uncertain: {summary['uncertain_cases']}")
            
        except Exception as e:
            print(f"  Error with {method_name}: {e}")
            results[method_name] = None
    
    # Save comparison summary
    comparison_file = os.path.join(output_dir, f"{image_name}_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\nComparison results saved to: {comparison_file}")
    
    return results


def batch_test(images_dir: str,
               detection_model_path: str,
               detection_config_path: Optional[str] = None,
               pose_model_path: Optional[str] = None,
               classifier_model_path: Optional[str] = None,
               output_dir: str = "./batch_results",
               max_images: Optional[int] = None) -> Dict:
    """
    Test the pipeline on multiple images
    """
    print(f"Batch testing images from: {images_dir}")
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))
        image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images to test")
    
    os.makedirs(output_dir, exist_ok=True)
    
    batch_results = {
        "total_images": len(image_files),
        "successful_tests": 0,
        "failed_tests": 0,
        "aggregate_stats": {
            "total_workers": 0,
            "workers_with_helmet": 0,
            "workers_without_helmet": 0,
            "uncertain_cases": 0
        },
        "detailed_results": []
    }
    
    for i, image_file in enumerate(image_files):
        print(f"\\nProcessing {i+1}/{len(image_files)}: {image_file.name}")
        
        try:
            result = test_single_image(
                image_path=str(image_file),
                detection_model_path=detection_model_path,
                detection_config_path=detection_config_path,
                pose_model_path=pose_model_path,
                classifier_model_path=classifier_model_path,
                output_dir=os.path.join(output_dir, "individual_results")
            )
            
            if result:
                batch_results["successful_tests"] += 1
                summary = result["enhancement_summary"]
                batch_results["aggregate_stats"]["total_workers"] += summary["total_workers"]
                batch_results["aggregate_stats"]["workers_with_helmet"] += summary["workers_with_helmet"]
                batch_results["aggregate_stats"]["workers_without_helmet"] += summary["workers_without_helmet"]
                batch_results["aggregate_stats"]["uncertain_cases"] += summary["uncertain_cases"]
                
                batch_results["detailed_results"].append({
                    "image": image_file.name,
                    "summary": summary,
                    "workers_count": len(result["workers"])
                })
            else:
                batch_results["failed_tests"] += 1
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            batch_results["failed_tests"] += 1
    
    # Save batch results
    batch_file = os.path.join(output_dir, "batch_summary.json")
    with open(batch_file, 'w') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print(f"\\n{'='*50}")
    print("BATCH TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Total images processed: {batch_results['total_images']}")
    print(f"Successful: {batch_results['successful_tests']}")
    print(f"Failed: {batch_results['failed_tests']}")
    print(f"\\nAggregate Statistics:")
    stats = batch_results["aggregate_stats"]
    print(f"  Total workers detected: {stats['total_workers']}")
    if stats['total_workers'] > 0:
        helmet_rate = stats['workers_with_helmet'] / stats['total_workers'] * 100
        no_helmet_rate = stats['workers_without_helmet'] / stats['total_workers'] * 100
        uncertain_rate = stats['uncertain_cases'] / stats['total_workers'] * 100
        print(f"  With helmet: {stats['workers_with_helmet']} ({helmet_rate:.1f}%)")
        print(f"  Without helmet: {stats['workers_without_helmet']} ({no_helmet_rate:.1f}%)")
        print(f"  Uncertain: {stats['uncertain_cases']} ({uncertain_rate:.1f}%)")
    
    print(f"\\nDetailed results saved to: {batch_file}")
    
    return batch_results


def main():
    parser = argparse.ArgumentParser(description='Test Enhanced Helmet Detection Pipeline')
    parser.add_argument('--mode', choices=['single', 'compare', 'batch'], required=True,
                       help='Test mode: single image, method comparison, or batch processing')
    parser.add_argument('--image', type=str,
                       help='Path to single image (required for single and compare modes)')
    parser.add_argument('--images_dir', type=str,
                       help='Directory of images (required for batch mode)')
    parser.add_argument('--detection_model', type=str, required=True,
                       help='Path to detection model (.pth file)')
    parser.add_argument('--detection_config', type=str,
                       help='Path to detection model config (.py file)')
    parser.add_argument('--pose_model', type=str,
                       help='Path to YOLO Pose model (.pt file)')
    parser.add_argument('--classifier_model', type=str,
                       help='Path to trained helmet classifier (.pth file)')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory for results')
    parser.add_argument('--max_images', type=int,
                       help='Maximum number of images to process in batch mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['single', 'compare'] and not args.image:
        print("Error: --image is required for single and compare modes")
        sys.exit(1)
    
    if args.mode == 'batch' and not args.images_dir:
        print("Error: --images_dir is required for batch mode")
        sys.exit(1)
    
    # Run appropriate test mode
    if args.mode == 'single':
        test_single_image(
            image_path=args.image,
            detection_model_path=args.detection_model,
            detection_config_path=args.detection_config,
            pose_model_path=args.pose_model,
            classifier_model_path=args.classifier_model,
            output_dir=args.output_dir
        )
        
    elif args.mode == 'compare':
        compare_methods(
            image_path=args.image,
            detection_model_path=args.detection_model,
            detection_config_path=args.detection_config,
            pose_model_path=args.pose_model,
            classifier_model_path=args.classifier_model,
            output_dir=args.output_dir
        )
        
    elif args.mode == 'batch':
        batch_test(
            images_dir=args.images_dir,
            detection_model_path=args.detection_model,
            detection_config_path=args.detection_config,
            pose_model_path=args.pose_model,
            classifier_model_path=args.classifier_model,
            output_dir=args.output_dir,
            max_images=args.max_images
        )


if __name__ == "__main__":
    main()