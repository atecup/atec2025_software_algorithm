# Hybrid YOLO-VLM Rescue Agent README

## Overview
This agent is designed to solve rescue missions in a simulated environment by combining object detection (YOLO) and vision-language models (VLM) to create a two-tiered decision system. The agent navigates through the environment, finds an injured person, and transports them to a stretcher or ambulance.

The agent successfully completes rescue tasks by implementing a hybrid approach:
1. **Primary Tier: YOLO Detection** - Provides precise object detection and movement when targets are clearly visible
2. **Secondary Tier: VLM Landmark Navigation** - Takes over when YOLO detection fails, using contextual understanding to search for landmarks

## System Architecture

### Two-Tier Decision System
The agent implements a hierarchical decision-making process:

1. **YOLO Object Detection (Primary)**
   - Runs first on each frame to detect key objects (person, stretcher, truck/ambulance)
   - Provides precise positioning and movement when objects are detected
   - Handles direct movement control with defined thresholds for actions

2. **VLM Landmark Navigation (Secondary)**
   - Activates when YOLO fails to detect relevant objects
   - Analyzes textual clues and visual environment
   - Uses landmark-based navigation to systematically search the environment

### YOLO Detection System
- **Object Detection**: Uses YOLO model to detect persons, stretchers (suitcases), trucks, and buses with confidence thresholds
- **Precision Control**: 
  - Divides screen into navigation zones (left, center, right)
  - Uses object coordinates to determine optimal movement action
  - Implements proximity thresholds for pickup and drop actions
- **State Tracking**: Maintains state of rescue operation (whether person has been picked up)

### Landmark-Based Navigation (VLM)
The agent parses the initial text clue to identify potential landmarks that might lead to the injured person. For example, if the clue is "The injured person is by the roadside garbage bin, and there is a car nearby," landmarks might include [roadside, car, garbage bin, injured person]. The agent will detect them in priority order and move towards them.

### Action and Observation Buffers
- **Action Buffer**: Maintains a queue of pending actions to be executed, allowing the agent to plan multiple steps ahead.
- **Observation Buffer**: Coordinates with the action buffer to determine which observations should be saved for later analysis.

These two buffers allow the VLM to perform continuous action control on the agent during the loop process.

## Technical Implementation Details

### YOLO Detection Logic
- **Confidence Thresholds**: Adapts confidence level based on whether a person has been picked up (0.1 when carrying, 0.2 when searching)
- **Object Prioritization**: Hierarchical detection logic that prioritizes persons when searching and stretcher/vehicles when carrying
- **Position-Based Actions**:
  - Central region (220-420 px): Move forward
  - Left region (<220 px): Turn left
  - Right region (>420 px): Turn right
  - Proximity triggers (y-coordinate thresholds): Initiate carry/drop actions

### VLM Navigation Features
- **Obstacle Avoidance**: The agent alternates between walking and jumping to navigate around obstacles when searching.
- **Repeated Pickup Attempts**: After finding the target person, the agent attempts to pick them up after each move.
- **Optimal Placement Distance**: When placing the injured person on the stretcher, the agent backs up before dropping.
- **Multi-Directional Observation**: The agent captures images from multiple directions and concatenates them.
- **Visual Guidance Lines**: When moving forward, the agent divides the image into regions using vertical lines.

## Execution Flow
1. The agent initializes and processes the initial clue.
2. On each frame, YOLO detection is attempted first:
   - If key objects are detected, precise movement actions are calculated based on their position.
   - If no relevant objects are detected, control passes to the VLM system.
3. The VLM system manages search, rescue, return, and placement phases:
   - Analyzes surroundings to locate landmarks from the clue
   - Moves toward identified landmarks
   - Picks up the injured person when found
   - Searches for the stretcher/ambulance
   - Places the injured person on the stretcher
