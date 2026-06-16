# RL-Miro Enhanced (Reinforcement Learning Maze-Solving Simulator)

RL-Miro Enhanced is an **educational reinforcement learning simulation program** with a Pygame-based GUI. It visualizes the trial-and-error process of an agent discovering the optimal path through a grid maze.

---

## 🌟 Key Features

1. **Multiple Reinforcement Learning Algorithms**:
   * **Q-Learning**: An off-policy algorithm aiming at the ideal shortest path.
   * **SARSA**: An on-policy algorithm incorporating exploration risk to prioritize safer paths.
   * **Double Q-Learning**: Uses decoupled Q-tables to eliminate the overestimation bias for stable convergence.
2. **Dynamic Grid Size Adjustment (5×5 to 30×30)**:
   * Real-time grid resize via dashboard slider.
   * Graphics (agent circle, special tile markings, borders) automatically scale proportionally to adapt to high-density grids.
3. **Maze Editor & DFS Generator**:
   * **Editor Mode**: Hand-draw walls, paths, traps, bonuses, starts, and goals with mouse clicks and drags. (Right-click to erase)
   * **🎲 Random Maze Generator**: Builds a solvable, random maze on-the-fly using the Depth-First Search (DFS) backtracking algorithm.
4. **Special Tile Mechanics**:
   * **Traps (Red X)**: Triggers a high penalty (-200) and immediately fails/restarts the episode (can be toggled ON/OFF in the control panel).
   * **Bonuses (Blue Star)**: Grants extra rewards (+150) with one-time collection logic per episode.
5. **Real-time Training Metrics & Graph**:
   * Keeps track of metrics like episode counts, success rates, best steps, and exploration probability (epsilon).
   * Visualizes a line graph plotting the steps taken over the last 100 episodes at the bottom of the sidebar.
6. **Automated Session Logging (`rl_training.log`)**:
   * Appends CSV formatted statistics (Episode, Steps, Success Rate, Parameters, Algorithm, Grid Size) for both training and testing sessions upon completion.
7. **Model Saving & Loading**:
   * Uses OS standard file dialogs (`tkinter`) to save/load custom maps and trained Q-tables in a compressed `.npz` format.

---

## 📁 Project Directory Structure (`src/`)

The application code is fully modularized for readability and clean decoupling.
```
src/
├── main.py             # Entry point of the application
├── rl/
│   ├── __init__.py
│   ├── agent.py        # 3 algorithms and Q-table data structures
│   └── environment.py  # Rewards, state transitions, and maze rules
├── ui/
│   ├── __init__.py
│   ├── gui.py          # Pygame GUI rendering and control loop
│   └── widgets.py      # Reusable UI widgets (Button, Slider)
└── utils/
    ├── __init__.py
    └── generator.py    # DFS backtracking maze generator utility
```

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.14.3 or higher**
* Pygame-CE, NumPy, Matplotlib, Seaborn (see `requirements.txt`)

### Running the Simulator
The project includes a batch file that automatically handles virtual environment creation, package installation, and execution.

```powershell
# On Windows
.\run.bat
```

Alternatively, you can activate the environment and run the commands manually:
```powershell
# Activate the virtual environment
.\miro_env\Scripts\activate.bat

# Install requirements
pip install -r requirements.txt

# Run the app
python src/main.py
```

---

## ⌨️ Hotkeys

* **`Q`**: Toggle the Q-value direction arrows (overlay) on/off in the grid window.
