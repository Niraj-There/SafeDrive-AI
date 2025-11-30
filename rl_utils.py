"""
Reinforcement Learning Utility Functions

Contains helper functions for state extraction and reward calculation
for the RL safety agent.
"""
from typing import Tuple, List, Optional
import numpy as np
from config import (
    ACTION_CRUISE, ACTION_WARN, ACTION_EMERGENCY,
    TRAFFIC_DENSITY_THRESHOLDS, PROXIMITY_THRESHOLDS
)


def extract_state_from_yolo(yolo_result, frame_idx: int, 
                            total_frames: int) -> Tuple[int, int, int]:
    """
    Extract state representation from YOLO detection results.
    
    State = (traffic_density, proximity_level, temporal_phase)
    
    Args:
        yolo_result: YOLO detection result object
        frame_idx: Current frame index
        total_frames: Total number of frames in video
    
    Returns:
        Tuple of (traffic_density: 0-2, proximity: 0-2, temporal_phase: 0-2)
        - traffic_density: 0=Low, 1=Medium, 2=High
        - proximity: 0=Safe, 1=Close, 2=Dangerous
        - temporal_phase: 0=Early, 1=Mid, 2=Late (position in video)
    """
    # Default state if no detections
    if not yolo_result.boxes or len(yolo_result.boxes) == 0:
        temporal_phase = get_temporal_phase(frame_idx, total_frames)
        return (0, 0, temporal_phase)
    
    # Calculate traffic density
    num_objects = len(yolo_result.boxes)
    if num_objects <= TRAFFIC_DENSITY_THRESHOLDS['low']:
        traffic_density = 0  # Low
    elif num_objects <= TRAFFIC_DENSITY_THRESHOLDS['medium']:
        traffic_density = 1  # Medium
    else:
        traffic_density = 2  # High
    
    # Calculate proximity (minimum distance between objects)
    proximity = calculate_proximity_level(yolo_result)
    
    # Calculate temporal phase
    temporal_phase = get_temporal_phase(frame_idx, total_frames)
    
    return (traffic_density, proximity, temporal_phase)


def calculate_proximity_level(yolo_result) -> int:
    """
    Calculate proximity level based on minimum distance between detected objects.
    
    Args:
        yolo_result: YOLO detection result
    
    Returns:
        0=Safe, 1=Close, 2=Dangerous
    """
    if not yolo_result.boxes or len(yolo_result.boxes) < 2:
        return 0  # Safe (not enough objects to compare)
    
    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    
    # Calculate centroids
    centroids = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centroids.append((cx, cy))
    
    # Find minimum distance between any two objects
    min_distance = float('inf')
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            cx1, cy1 = centroids[i]
            cx2, cy2 = centroids[j]
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            min_distance = min(min_distance, distance)
    
    # Classify proximity
    if min_distance <= PROXIMITY_THRESHOLDS['dangerous']:
        return 2  # Dangerous
    elif min_distance <= PROXIMITY_THRESHOLDS['close']:
        return 1  # Close
    else:
        return 0  # Safe


def get_temporal_phase(frame_idx: int, total_frames: int) -> int:
    """
    Determine temporal phase of current frame in video.
    
    Args:
        frame_idx: Current frame index
        total_frames: Total frames in video
    
    Returns:
        0=Early (first third), 1=Mid (middle third), 2=Late (last third)
    """
    if total_frames == 0:
        return 1  # Default to mid
    
    progress = frame_idx / total_frames
    
    if progress < 0.33:
        return 0  # Early
    elif progress < 0.67:
        return 1  # Mid
    else:
        return 2  # Late


def calculate_reward(action: int, ground_truth_incident: int, 
                    is_incident_frame: bool, proximity: int,
                    traffic_density: int) -> float:
    """
    Calculate reward based on action, ground truth, and context.
    
    Reward Structure:
    - Correct emergency action during collision: +100
    - Correct warning during near collision: +50
    - Correct cruise in safe conditions: +10
    - Wrong action during danger: -50 to -100
    - Unnecessary emergency action: -20
    
    Args:
        action: Agent's chosen action (0=CRUISE, 1=WARN, 2=EMERGENCY)
        ground_truth_incident: Ground truth label (0=No, 1=Near, 2=Collision)
        is_incident_frame: Whether this is the incident frame
        proximity: Proximity level (0=Safe, 1=Close, 2=Dangerous)
        traffic_density: Traffic density (0=Low, 1=Medium, 2=High)
    
    Returns:
        Reward value (float)
    """
    reward = 0.0
    
    # COLLISION SCENARIO
    if ground_truth_incident == 2:  # Collision
        if is_incident_frame or proximity == 2:
            if action == ACTION_EMERGENCY:
                reward = 100.0  # Excellent! Emergency brake during collision
            elif action == ACTION_WARN:
                reward = 20.0   # Warning is better than nothing
            else:
                reward = -100.0  # Failed to react to collision!
        else:
            # Before collision
            if action == ACTION_WARN:
                reward = 30.0   # Good anticipation
            elif action == ACTION_EMERGENCY:
                reward = 10.0   # Too early but safe
            else:
                reward = 5.0    # Neutral
    
    # NEAR COLLISION SCENARIO
    elif ground_truth_incident == 1:  # Near collision
        if is_incident_frame or proximity >= 1:
            if action == ACTION_WARN:
                reward = 50.0   # Perfect! Warning for near collision
            elif action == ACTION_EMERGENCY:
                reward = 20.0   # Too aggressive but safe
            else:
                reward = -50.0  # Failed to warn about danger
        else:
            # Safe region in near-collision video
            if action == ACTION_CRUISE:
                reward = 10.0
            elif action == ACTION_WARN:
                reward = 5.0
            else:
                reward = -10.0  # Unnecessary emergency
    
    # NO INCIDENT SCENARIO
    else:  # No incident
        if proximity == 2:  # But objects are dangerously close
            if action == ACTION_WARN or action == ACTION_EMERGENCY:
                reward = 20.0   # Good defensive driving
            else:
                reward = -10.0  # Should be more cautious
        elif proximity == 1:  # Objects are close
            if action == ACTION_WARN:
                reward = 15.0   # Good caution
            elif action == ACTION_CRUISE:
                reward = 5.0    # Acceptable
            else:
                reward = -5.0   # Too aggressive
        else:  # Safe conditions
            if action == ACTION_CRUISE:
                reward = 10.0   # Perfect! Cruise in safe conditions
            elif action == ACTION_WARN:
                reward = 0.0    # Overly cautious
            else:
                reward = -20.0  # Unnecessary emergency brake
    
    # Bonus for high traffic density awareness
    if traffic_density == 2 and action != ACTION_CRUISE:
        reward += 5.0  # Bonus for caution in heavy traffic
    
    return reward


def get_action_name(action: int) -> str:
    """
    Convert action index to readable name.
    
    Args:
        action: Action index
    
    Returns:
        Action name string
    """
    action_names = {
        ACTION_CRUISE: 'CRUISE',
        ACTION_WARN: 'WARN',
        ACTION_EMERGENCY: 'EMERGENCY'
    }
    return action_names.get(action, 'UNKNOWN')


def get_incident_name(incident: int) -> str:
    """
    Convert incident index to readable name.
    
    Args:
        incident: Incident index
    
    Returns:
        Incident name string
    """
    incident_names = {
        0: 'No Incident',
        1: 'Near Collision',
        2: 'Collision'
    }
    return incident_names.get(incident, 'UNKNOWN')
