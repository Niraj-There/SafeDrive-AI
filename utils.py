"""Utility functions for video processing and data handling

Contains helper functions for loading videos, preprocessing frames,
and validating inputs.
"""
from typing import Tuple, List, Optional, Dict
import cv2
import torch
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_video_file(video_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a video file exists and is readable.
    
    Args:
        video_path (str): Path to video file
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    path = Path(video_path)
    if not path.exists():
        return (False, f'Video file does not exist: {video_path}')
    
    if not path.is_file():
        return (False, f'Path is not a file: {video_path}')
    
    # Try to open with OpenCV to verify it's a valid video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return (False, f'Cannot open video file: {video_path}')
    
    cap.release()
    return (True, None)


def load_video_frames(video_path: str, max_frames: Optional[int] = None, 
                     sample_rate: int = 1) -> Tuple[List[np.ndarray], bool]:
    """
    Load frames from a video file with memory management.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (prevents memory overflow)
        sample_rate: Take every nth frame (1 = all frames)
    
    Returns:
        Tuple of (frames list, success boolean)
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return (frames, False)
    
    try:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Warn if loading too many frames
        if max_frames is None and total_frames > 1000:
            logger.warning(f"Loading {total_frames} frames may consume significant memory")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
    except Exception as e:
        logger.error(f"Error loading frames: {e}")
        return (frames, False)
    finally:
        cap.release()
    
    return (frames, len(frames) > 0)


def preprocess_frames(frames: List[np.ndarray], target_size: Tuple[int, int] = (224, 224), 
                     normalize: bool = True) -> torch.Tensor:
    """
    Preprocess video frames for model input.
    
    Args:
        frames: List of numpy arrays (RGB images)
        target_size: Target image size (height, width)
        normalize: Whether to normalize pixel values using ImageNet stats
    
    Returns:
        Preprocessed frames as tensor [batch, channels, height, width]
    
    Raises:
        ValueError: If frames list is empty or contains invalid data
    """
    if not frames:
        raise ValueError('No frames provided for preprocessing')
    
    if not all(isinstance(f, np.ndarray) for f in frames):
        raise ValueError('All frames must be numpy arrays')
    
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    
    processed_frames = []
    for frame in frames:
        processed = transform(frame)
        processed_frames.append(processed)
    
    # Stack into tensor [batch, channels, height, width]
    return torch.stack(processed_frames)


def calculate_traffic_density(detected_objects: Dict[str, int]) -> int:
    """
    Calculate traffic density based on detected objects.
    
    Args:
        detected_objects: Dictionary of {class_name: count}
    
    Returns:
        Traffic density level (1=Low, 2=Medium, 3=High)
    """
    from config import TRAFFIC_DENSITY_THRESHOLDS
    
    if not detected_objects:
        return 1  # Low - no objects detected
    
    total_objects = sum(detected_objects.values())
    
    if total_objects <= TRAFFIC_DENSITY_THRESHOLDS['low']:
        return 1  # Low
    elif total_objects <= TRAFFIC_DENSITY_THRESHOLDS['medium']:
        return 2  # Medium
    else:
        return 3  # High


def calculate_proximity(object_positions: List[Tuple[float, float]], 
                       ego_position: Tuple[float, float] = (0, 0),
                       threshold_close: Optional[int] = None,
                       threshold_dangerous: Optional[int] = None) -> int:
    """
    Calculate proximity level based on object distances from ego vehicle.
    
    Args:
        object_positions: List of (x, y) positions
        ego_position: Position of ego vehicle (default: origin)
        threshold_close: Distance threshold for "close" (uses config if None)
        threshold_dangerous: Distance threshold for "dangerous" (uses config if None)
    
    Returns:
        Proximity level (0=Safe, 1=Close, 2=Dangerous)
    """
    from config import PROXIMITY_THRESHOLDS
    
    # Use config defaults if not provided
    if threshold_close is None:
        threshold_close = PROXIMITY_THRESHOLDS['close']
    if threshold_dangerous is None:
        threshold_dangerous = PROXIMITY_THRESHOLDS['dangerous']
    
    if not object_positions:
        return 0  # Safe (no objects nearby)
    
    # Calculate distances from ego position
    distances = []
    for pos in object_positions:
        dist = np.sqrt((pos[0] - ego_position[0])**2 + (pos[1] - ego_position[1])**2)
        distances.append(dist)
    
    min_distance = min(distances)
    
    if min_distance <= threshold_dangerous:
        return 2  # Dangerous
    elif min_distance <= threshold_close:
        return 1  # Close
    else:
        return 0  # Safe


def save_results_to_csv(results: List[Dict], output_path: str) -> None:
    """
    Save results to CSV file.
    
    Args:
        results: List of dictionaries with results
        output_path: Path to output CSV file
    
    Raises:
        IOError: If file cannot be written
    """
    import pandas as pd
    
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f'[SUCCESS] Results saved to {output_path}')
    except Exception as e:
        logger.error(f'[ERROR] Failed to save results: {e}')
        raise IOError(f'Could not write to {output_path}: {e}')
