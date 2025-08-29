import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, efficientnet_b0
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


class HelmetClassifier(nn.Module):
    def __init__(self, 
                 backbone: str = "resnet18", 
                 num_classes: int = 2, 
                 pretrained: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "resnet18":
            m = resnet18(pretrained=pretrained)
            feature_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        elif backbone == "resnet34":
            m = resnet34(pretrained=pretrained)
            feature_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        elif backbone == "efficientnet_b0":
            m = efficientnet_b0(pretrained=pretrained)
            feature_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            self.backbone = m
        else:
            if not _HAS_TIMM:
                raise ValueError(
                    f"Unsupported backbone: {backbone}. "
                    f"Install timm to use additional backbones."
                )
            try:
                self.backbone = timm.create_model(
                    backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
                )
                feature_dim = self.backbone.num_features
            except Exception as e:
                raise ValueError(f"Unsupported backbone: {backbone}. timm error: {e}")
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        logits = self.classifier(feats)
        return logits


class HelmetClassificationInference:
    """
    Inference wrapper for helmet classification
    """
    
    def __init__(self, 
                 model_path: str, 
                 device: Optional[str] = None,
                 input_size: Tuple[int, int] = (224, 224)):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class labels
        self.class_names = ['no_helmet', 'wearing_helmet']
    
    def _load_model(self, model_path: str) -> HelmetClassifier:
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model parameters from checkpoint if available
            model_config = checkpoint.get('model_config', {})
            model = HelmetClassifier(
                backbone=model_config.get('backbone', 'resnet18'),
                num_classes=model_config.get('num_classes', 2),
                pretrained=False,
                dropout_rate=model_config.get('dropout_rate', 0.2)
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            print(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference
        Args:
            image: OpenCV image (BGR format)
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict helmet presence in image
        Args:
            image: OpenCV image (BGR format)
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to numpy
            predicted_class = predicted_class.cpu().item()
            confidence = confidence.cpu().item()
            probabilities = probabilities.cpu().numpy()[0]
            
            result = {
                'predicted_class': predicted_class,
                'class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'no_helmet': float(probabilities[0]),
                    'wearing_helmet': float(probabilities[1])
                },
                'wearing_helmet': predicted_class == 1
            }
            
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'predicted_class': -1,
                'class_name': 'error',
                'confidence': 0.0,
                'probabilities': {'no_helmet': 0.0, 'wearing_helmet': 0.0},
                'wearing_helmet': False
            }
    
    def predict_batch(self, images: list) -> list:
        """
        Batch prediction for multiple images
        Args:
            images: List of OpenCV images (BGR format)
        Returns:
            List of prediction results
        """
        results = []
        
        try:
            # Preprocess all images
            input_tensors = []
            for image in images:
                tensor = self.preprocess_image(image)
                input_tensors.append(tensor)
            
            # Batch inference
            batch_tensor = torch.cat(input_tensors, dim=0)
            
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = F.softmax(logits, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
            
            # Process results
            for i in range(len(images)):
                pred_class = predicted_classes[i].cpu().item()
                confidence = confidences[i].cpu().item()
                probs = probabilities[i].cpu().numpy()
                
                result = {
                    'predicted_class': pred_class,
                    'class_name': self.class_names[pred_class],
                    'confidence': confidence,
                    'probabilities': {
                        'no_helmet': float(probs[0]),
                        'wearing_helmet': float(probs[1])
                    },
                    'wearing_helmet': pred_class == 1
                }
                results.append(result)
            
        except Exception as e:
            print(f"Batch prediction error: {e}")
            # Return error results for all images
            error_result = {
                'predicted_class': -1,
                'class_name': 'error',
                'confidence': 0.0,
                'probabilities': {'no_helmet': 0.0, 'wearing_helmet': 0.0},
                'wearing_helmet': False
            }
            results = [error_result.copy() for _ in images]
        
        return results


def create_model_config(backbone: str = "resnet18", 
                       num_classes: int = 2, 
                       dropout_rate: float = 0.2) -> Dict[str, Any]:
    return {
        'backbone': backbone,
        'num_classes': num_classes,
        'dropout_rate': dropout_rate,
        'input_size': (224, 224),
        'class_names': ['no_helmet', 'wearing_helmet']
    }


if __name__ == "__main__":
    # Test model creation
    model = HelmetClassifier(backbone="resnet18", num_classes=2)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")