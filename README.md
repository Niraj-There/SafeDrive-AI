# SafeDrive-AI: Autonomous Accident Detection & Avoidance System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ADD8.svg)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

> **Capstone Project: Reinforcement Learning with Agentic AI & AI Ethics**  
> *Department of Computer Science & Engineering*

---

## 📑 Abstract

**SafeDrive-AI** represents a novel approach to autonomous vehicle safety systems by integrating computer vision with reinforcement learning (RL). Unlike traditional rule-based ADAS (Advanced Driver Assistance Systems), this project leverages a hybrid architecture: a **ResNet-18** backbone for temporal accident classification, **YOLOv8** for real-time object tracking, and a **Q-Learning** agent for adaptive decision-making. The system is designed to not only detect hazardous situations but to autonomously learn optimal intervention strategies (Warn, Brake, Cruise) based on dynamic traffic density and proximity metrics. This work also addresses critical **AI Ethics** concerns, focusing on transparency and safety in autonomous decision loops.

---

## 🏗️ System Architecture

The system operates on a closed-loop control pipeline consisting of three primary modules:

### 1. Perception Module (Object Tracking)
- **Algorithm**: YOLOv8m (Medium) + ByteTrack
- **Function**: Real-time detection and tracking of dynamic agents (Vehicles, Pedestrians, Animals).
- **Output**: Bounding boxes, Class IDs, and Trajectory vectors.

### 2. Classification Module (Incident Detection)
- **Architecture**: ResNet-18 (Pre-trained on ImageNet, fine-tuned)
- **Input**: Temporal stack of 24 frames ($T=24$).
- **Classes**: `No Incident`, `Near Collision`, `Collision`.
- **Mechanism**: Spatial feature extraction followed by temporal pooling to capture accident dynamics.

### 3. Decision Module (Agentic AI)
- **Algorithm**: Tabular Q-Learning (Model-Free RL).
- **State Space ($S$)**: Discretized tuple of `(Traffic_Density, Proximity_Level)`.
- **Action Space ($A$)**:
  - `CRUISE` ($a=0$): Maintain velocity.
  - `WARN` ($a=1$): Issue driver alert.
  - `EMERGENCY` ($a=2$): Autonomous braking/evasion.
- **Reward Function ($R$)**: Heavily penalized for collisions; rewarded for successful avoidance and smooth driving.

---

## 🛠️ Technical Implementation

### Prerequisites
- **OS**: Windows / Linux / MacOS
- **Python**: 3.8+
- **Compute**: CUDA-enabled GPU recommended (tested on NVIDIA RTX series).

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Niraj-There/SafeDrive-AI.git
    cd SafeDrive-AI
    ```

2.  **Environment Setup**
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Model Acquisition**
    Download the pre-trained weights for the accident classifier and YOLOv8.
    ```bash
    python download_models.py
    ```

---

## 💻 Usage & Reproducibility

### Experiment 1: Batch Video Analysis
Run the perception and classification pipeline on a dataset of dashcam footage.
```bash
python batch_video_analysis.py
```
*Output*: Generates `submission.csv` containing frame-level incident predictions and severity classifications.

### Experiment 2: Object Tracking Visualization
Visualize the YOLOv8 tracking and movement analysis on a specific sample.
```bash
python yolo_object_tracker.py --video Video/sample.mp4 --output yolo_output/result.mp4
```

### Experiment 3: RL Agent Training
Train the Q-Learning agent within the video environment.
```bash
python rl_training.py
```
*Process*:
1.  Iterates through video episodes.
2.  Extracts state vectors from visual data.
3.  Updates Q-Table using the Bellman Equation:
    $$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s,a)]$$
4.  Saves the converged policy to `q_table.pkl`.

---

## 📊 Performance Metrics

| Module | Metric | Value |
| :--- | :--- | :--- |
| **Object Detection** | mAP@0.5 | 0.76 |
| **Accident Classification** | Accuracy | ~85% |
| **Inference Speed** | FPS (GPU) | 30-45 |
| **RL Convergence** | Avg. Reward | >3500 (Ep 3) |

---

## ⚖️ AI Ethics & Safety Considerations

In alignment with the **Capstone Project on AI Ethics**, this system incorporates:
-   **Fail-Safe Design**: The `EMERGENCY` action is prioritized in high-certainty collision states.
-   **Interpretability**: The Q-Table provides a transparent mapping of states to actions, allowing for auditability of the decision logic.
-   **Bias Mitigation**: The dataset includes diverse traffic scenarios to prevent overfitting to specific road conditions.

---

## 🔮 Future Research Directions

1.  **Deep Q-Networks (DQN)**: Transitioning from tabular Q-learning to Deep RL to handle continuous state spaces.
2.  **Multi-Modal Fusion**: Integrating LiDAR and Radar data for robust depth estimation.
3.  **V2X Communication**: Enabling vehicle-to-everything communication for cooperative collision avoidance.

---

## 👨‍💻 Author

**Niraj There**  
*B.Tech Computer Science & Engineering*  
*Specialization in Artificial Intelligence & Machine Learning*

---

*This project is licensed under the MIT License.*
