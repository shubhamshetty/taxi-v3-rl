# Taxi-v3-rl
Reinforcement Learning agents navigating Taxi-v3: Q-Learning vs SARSA vs DQN

## 🤖 Algorithms Implemented
- **Q-Learning** (Off-policy)
- **SARSA** (On-policy)
- **DQN (Deep Q-Network)** using PyTorch

## 🎮 Environment
- **Taxi-v3** from OpenAI Gym
- Grid: 5x5 map, 4 passenger locations
- State space: 500 discrete states
- Action space: 6 discrete actions

## 🧪 Results Overview

| Algorithm | Convergence | Avg. Reward | Efficiency |
|-----------|-------------|-------------|------------|
| DQN       | ~2000 ep    | Highest     | Most Efficient |
| SARSA     | ~2500 ep    | Moderate    | Balanced |
| Q-Learning| ~3000 ep    | Lower       | Slowest |


## 🛠 Setup

### ✅ Install Dependencies
```bash
pip install -r requirements.txt

# Run Agent Evaluation (Change the model type in src/eval.py to one of: "DQN", "SARSA", or "Q-Learning" and run):
python src/eval.py
```

## 🚀 Future Work
- Implement advanced DQNs (Double DQN, Dueling DQN)
- Multi-agent taxi simulation
- Real-world scenario adaptation using transfer learning


