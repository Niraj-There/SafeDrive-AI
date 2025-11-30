"""Model loading and architecture definitions
Contains functions to load trained models for accident detection.
"""
from typing import Optional, Union
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AccidentDetectionModel(nn.Module):
    """
    CNN-based accident detection model (VideoClassifier architecture).
    Uses ResNet18 base with custom classifier.
    Classifies videos into: No Incident, Near Collision, or Collision.
    """
    
    def __init__(self, num_classes=3):
        super(AccidentDetectionModel, self).__init__()
        from torchvision import models as tv_models
        base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.base = base
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Handle video input: [B, T, C, H, W]
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feats = self.base(x)
            feats = feats.view(B, T, 512).mean(1)
        else:
            # Single image input: [B, C, H, W]
            feats = self.base(x)
        out = self.fc(feats)
        return out


def load_accident_model(model_path: str, device: str = 'cpu', 
                       num_classes: int = 3) -> nn.Module:
    """
    Load a trained accident detection model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pth file)
        device: Device to load model on ('cuda' or 'cpu')
        num_classes: Number of output classes (default: 3)
    
    Returns:
        Loaded PyTorch model in eval mode
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f'Model file not found: {model_path}')
    
    logger.info(f'Loading model from {model_path}')
    
    try:
        model = AccidentDetectionModel(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        logger.info(f'[SUCCESS] Model loaded successfully on {device}')
        return model
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        raise RuntimeError(f'Failed to load model from {model_path}: {e}')


def load_rl_agent(q_table_path: str) -> dict:
    """
    Load a trained Q-Learning agent's Q-table.
    Args:
        q_table_path: Path to the Q-table pickle file
    
    Returns:
        Q-table dictionary mapping states to action values
    
    Raises:
        FileNotFoundError: If Q-table file doesn't exist
        RuntimeError: If loading fails
    """
    import pickle
    
    q_table_path_obj = Path(q_table_path)
    if not q_table_path_obj.exists():
        raise FileNotFoundError(f'Q-table file not found: {q_table_path}')
    
    logger.info(f'Loading Q-table from {q_table_path}')
    
    try:
        with open(q_table_path, 'rb') as f:
            q_table = pickle.load(f)
        logger.info(f'[SUCCESS] Q-table loaded with {len(q_table)} states')
        return q_table
    except Exception as e:
        logger.error(f'Failed to load Q-table: {e}')
        raise RuntimeError(f'Failed to load Q-table from {q_table_path}: {e}')
