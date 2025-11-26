# VirtualRobotics (Autonomous Agent in Unity)

> Main idea: To create a simulation platform in the Unity engine for testing autonomous navigation and Computer Vision (CV) algorithms. The project involves building a "virtual robot" (agent) that makes movement decisions based on analysis of the virtual camera image, eliminating the need to build and test expensive physical hardware.

---

## Goal and Motivation

- **To gain experience:** The opportunity to work on advanced topics at the intersection of AI, CV and robotics in a controlled, virtual environment.
- **Low cost of entry:** Avoiding the costs and logistical problems associated with building physical robots (cost of sensors, hardware failure, time-consuming testing).
- **Rapid prototyping:** Ease of testing different models and scenarios without physical constraints.
- **Visual appeal:** Creating a "playable" design that is easy to present (e.g., at open days, scientific conferences).

---

## System Architecture

The project will consist of three main, cooperating modules:

### 1. simulation environment (Unity engine)

- **Task: To** create a virtual world in which the agent will operate.
- **Components:**
    - **Map:** 3D (or 2D) environment containing targets, obstacles and the agent.
    - **Agent (Character):** An object that has physics and "actuators" (movement system).
    - **Virtual Camera:** The agent's sensor, rendering an image from the agent's perspective.
    - **Control API:** A simple script (C#) that allows you to send commands to the agent (e.g. `MoveForward()`, `Rotate(angle)`).

### 2nd Perception Module (Computer Vision).

- **Task:** Analyze a "video stream" from a virtual camera and interpret what the agent "sees".
- **Components:**
    - **Video capture:** Receiving frames from the Unity camera (e.g. via texture streaming or API).
    - **Pre-processing:** Image conversion, noise reduction.
    - **CV model:** Responsible for object recognition (e.g., "red square," "wall," "obstacle").
    - **Output:** Structured data for the Decision Module (e.g. `{"object": "red_square", "position": [x, y], "distance": z}`).

### 3 Decision Module (AI / Control).

- **Task:** Decide on the agent's next move based on data from the Perception Module and the mission objective.
- **Components:**
    - **Input:** Mission objective (e.g. `FIND(red_square)`) and data from the Perception Module.
    - **Decision logic: The** algorithm that controls the agent.
        - *Simple version:* State automaton (e.g., `if target_visible: move_towards; else: scan_area`).
        - *Advanced version:* Pathfinding algorithms (e.g., A*), Reinforcement Learning.
    - **Output:** Commands sent to the Control API in Unity (e.g. `Rotate(15)`).

---

## Test Scenario (Proof of Concept).

A description of a minimum viable product (MVP) that will validate the assumptions.

- **Mission Objective:** The agent receives the command: `GO_TO(red_square)`.
- **Environment:** A simple, empty room (map) with three objects on the floor in different locations:
    1. Red square
    2. Blue circle
    3. Green triangle
- **Action Flow (Workflow):**
    1. Agent starts at a random point.
    2. The agent rotates in place (state: `SCANNING`), analyzing the environment frame by frame.
    3. **A CV Module** (such as a simple OpenCV script) analyzes the image for color and shape.
    4. When the CV Module detects a "red square," it transmits its position on the screen to **the Decision Module**.
    5. **The Decision Module** goes into a `NAVIGATING` state:
        - If the target is on the left side of the screen -> send the `Rotate(-5)` command.
        - If the target is on the right side of the screen -> send the `Rotate(5)` command.
        - If the target is in the center of the screen -> send `MoveForward()` command.
    6. The task succeeds when the agent comes within a certain distance from the target (e.g., based on collision or distance).

---

## Potential Project Expansions

- **Obstacle avoidance:** Adding static and dynamic obstacles on the map that the agent must avoid.
- **Reinforcement Learning (RL):** Instead of writing decision logic by hand, an agent can be trained to learn to navigate by trial and error on its own (rewarding it for reaching its destination).
- **More complex tasks:** E.g., "Find all the green objects" or "Bring me the blue ball."
- **Swarm simulation:** Control multiple agents that have to cooperate with each other.