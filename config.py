"""
Configuration file for Accident Detection System

Contains paths, device settings, and constants used across the project.
"""
import torch

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model paths
ACCIDENT_MODEL_PATH = 'accident_model.pth'
YOLO_MODEL_PATH = 'yolov8m.pt'
Q_TABLE_PATH = 'q_table.pkl'

# Label mappings
INCIDENT_LABEL_MAP = {
    0: 'No Incident',
    1: 'Near Collision',
    2: 'Collision'
}

# YOLO allowed classes
ALLOWED_CLASSES = ['car', 'truck', 'bus', 'person', 'dog', 'cat', 'cow']

# Default paths
DEFAULT_VIDEO_PATH = 'Video/525.mp4'
OUTPUT_VIDEO_PATH = 'yolo_output/tracked_output.mp4'

# Reinforcement Learning configuration
RL_ACTIONS = ['CRUISE', 'WARN', 'EMERGENCY']
ACTION_CRUISE = 0
ACTION_WARN = 1
ACTION_EMERGENCY = 2

# State space configuration
TRAFFIC_DENSITY_LEVELS = {1: 'Low', 2: 'Medium', 3: 'High'}
PROXIMITY_LEVELS = {0: 'Safe', 1: 'Close', 2: 'Dangerous'}

# Thresholds for state extraction
TRAFFIC_DENSITY_THRESHOLDS = {'low': 2, 'medium': 5}  # object counts
PROXIMITY_THRESHOLDS = {'close': 100, 'dangerous': 50}  # pixel distance
INCIDENT_FRAME_WINDOW = 15  # frames around incident to consider

# Video processing
NUM_FRAMES_SAMPLE = 24  # frames to sample for accident classification
DEFAULT_FRAME_SAMPLE_RATE = 1  # take every nth frame
MAX_FRAMES_IN_MEMORY = 300  # prevent memory overflow
