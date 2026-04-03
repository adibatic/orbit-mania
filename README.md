# Orbit Mania
**A Newtonian Space Simulation & Reinforcement Learning Environment**

## Overview

Orbit Mania is an arcade-style space simulation where players or AI agents pilot a spacecraft, dodge asteroids, and manage fuel using Hohmann transfers.

The project consists of a physics simulation built into a headless OpenAI `gymnasium` environment, which is decoupled from the Pygame graphics. This architecture supports training reinforcement learning agents using the Proximal Policy Optimization (PPO) algorithm.

---

## Repository Structure

```
orbit-mania/
├── src/
│   ├── env.py           # Gymnasium RL environment and physics engine
│   ├── play.py          # Pygame rendering engine and main entry point
│   ├── train.py         # AI script for training the PPO agent
│   └── utilities.py     # Graphics bridging and asset loaders
├── assets/              # Sprites, graphics, and sound assets
├── figures/             # LaTeX diagrams and scientific figures
├── paper/               # Scientific game report (LaTeX source)
├── models/              # Pretrained neural network weights
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Software Requirements

- **Host PC:** Python ≥ 3.10 (Linux / macOS / Windows)
- **Engine:** `pygame`, `numpy`
- **Machine Learning:** `gymnasium`, `stable-baselines3[extra]`
- **Documentation:** LaTeX distribution (TeX Live / MiKTeX) for compiling the physics report

---

## Installation & Setup

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

> **Note:** TensorBoard is automatically installed alongside `stable-baselines3` and can be used to view loss graphs during training.

---

## Usage

### Manual Play (Human Mode)

```bash
python src/play.py
```

**Controls:**

| Key | Action |
|-----|--------|
| `Space` | Switch orbital radius (inner `r=125` ↔ outer `r=225`) |
| `↑` / `↓` Arrow | Fine-tune prograde / retrograde thrust |
| `S` (hold) | Toggle deflector shield (consumes fuel) |
| `Esc` | Quit game |

### AI Inference Mode

Run a pretrained RL agent:

```bash
python src/play.py --ai --model models/ppo_orbit_mania.zip
```

### Training a New AI Agent

```bash
python src/train.py
```

Monitor training metrics with TensorBoard:

```bash
tensorboard --logdir logs/
```

---

## System Architecture

### 1. The Headless Environment (`env.py`)

The `OrbitEnv` class extends `gymnasium.Env`. It uses Euler integration over discrete time steps (`dt`) to calculate acceleration based on $F = G \frac{Mm}{r^2}$. The environment operates independently of the graphics, processing state arrays (Observation Space) and accepting inputs (Action Space).

### 2. Pygame Rendering Loop (`play.py`)

The Pygame wrapper reads the coordinate arrays produced by `OrbitEnv.step()` and projects them into 2D sprite transformations for visualization.

---

## Writing & Manuscript

The project manuscript is located in the `paper/` directory.

- Compile `paper/main.tex` using `latexmk` or an equivalent editor extension.
- Requires a LaTeX distribution such as TeX Live or MiKTeX.
- The manuscript pulls vector figures directly from the `figures/` directory.

---

## Author

**Adriel I. Santoso**  
Department of Mechanical and Aerospace Engineering, Tohoku University