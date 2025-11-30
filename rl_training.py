import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm
from rl_agent import QLearningAgent
from Both_mix_code_submition import VideoClassifier, sample_frames, transform, classify_incident, find_incident_frame
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VIDEO_DIR = 'Video'
ALLOWED_CLASSES = {'bicycle', 'bus', 'truck', 'person', 'car'}
ACTION_CRUISE = 0
ACTION_WARN = 1
ACTION_BRAKE = 2
ACTIONS = [ACTION_CRUISE, ACTION_WARN, ACTION_BRAKE]

def get_state(yolo_result, frame_idx):
    """
Discretize the environment into a simple state for the RL agent.
State = (Traffic_Density, Proximity_Alert)
"""
    if not yolo_result.boxes:
        pass
    return (0, 0)

def calculate_reward(action, ground_truth_incident, is_incident_frame):
    """
Reward function based on Agent Action vs Reality
ground_truth_incident: 0 (No), 1 (Near), 2 (Collision)
"""
    reward = 0
    if ground_truth_incident == 2 and is_incident_frame:
        if action == ACTION_BRAKE:
            reward = 100
            return reward

def train_rl_agent():
    print('🚀 Starting RL Agent Training on Video Data...')
    agent = QLearningAgent(actions=ACTIONS)
    yolo = YOLO('yolov8m.pt').to(DEVICE)
    accident_model = VideoClassifier().to(DEVICE)
    accident_model.load_state_dict(torch.load('accident_model.pth', map_location=DEVICE)) if os.path.exists('accident_model.pth') else None
    accident_model.eval()
    videos = [v for v in os.listdir(VIDEO_DIR) if v.endswith('.mp4')]
    EPISODES = 3
    for episode in range(EPISODES):
        total_reward = 0
        print(f'\n--- Episode {episode + 1}/{EPISODES} ---')
        for vid in tqdm(videos, desc='Training on Videos'):
            video_path = os.path.join(VIDEO_DIR, vid)
            all_frames = []
            yolo_results = []
            results = yolo.track(source=video_path, stream=True, tracker='bytetrack.yaml', conf=0.5, verbose=False, persist=True)
            for r in results:
                yolo_results.append(r)
                all_frames.append(cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB))
            pred_label = classify_incident(all_frames)
            incident_frame_idx = find_incident_frame(yolo_results) if pred_label != 0 else -1
            prev_state = None
            prev_action = None
            for i, result in enumerate(yolo_results):
                current_state = get_state(result, i)
                action = agent.choose_action(current_state)
                is_danger_zone = False
                is_danger_zone = True if incident_frame_idx != -1 and abs(i - incident_frame_idx) < 15 else False
                current_situation = pred_label if is_danger_zone else 0
                reward = calculate_reward(action, current_situation, is_danger_zone)
                total_reward += reward
                agent.learn(prev_state, prev_action, reward, current_state) if prev_state is not None else None
                prev_state = current_state
                prev_action = action
        print(f'Episode {episode + 1} Total Reward: {total_reward}')
        agent.save_model()
if __name__ == '__main__':
    train_rl_agent()