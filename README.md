# PaanaAI — Player Assessment and Next Action AI

PaanaAI (Player Assessment and Next Action Artificial Intelligence) is a tactical football analysis system that combines computer vision and spatial analytics to evaluate player roles and recommend optimal decisions on the pitch.

It analyzes 2D heatmaps of player activity to classify roles, and uses real-time positional data to compute the best next action, balancing risk and reward.

---

## Features

### Player Role Detection
- Upload a player heatmap
- CNN-based model classifies into roles:
  - Striker
  - Left Winger
  - Right Winger
  - Central Midfielder
  - Defensive Midfielder
  - Attacking Midfielder
  - Left Back
  - Right Back
  - Centre Back

- Outputs:
  - Predicted role
  - Confidence score

---

## Best Pass Recommendation

**Input:**
- Positions of both teams
- Ball carrier
- Game Mode:
    - Ultra Defensive
    - Defensive
    - Balanced
    - Attacking
    - Ultra Attacking


**Output:**
- Best pass target
- Pass type (ground / lofted)
- Alternative decision (shoot / keep the ball)

**Based on:**
- Expected Threat (xT)
- Defensive pressure
- Passing lanes and interception risk

---

## Core Concepts

- xT (Expected Threat) grid-based evaluation
- Risk vs Reward Optimization
- Passing Lane Geometry
- CNN-based Image Classification
- Tactical Decision Modeling