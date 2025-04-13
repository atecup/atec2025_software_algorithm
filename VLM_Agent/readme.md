# Landmark Agent README

## Overview

This agent is designed to solve rescue missions in a simulated environment by analyzing textual clues and using vision-language models (VLM) to navigate through the environment, find an injured person, and transport them to a stretcher or ambulance.

The agent successfully completes rescue tasks on some test scenarios by implementing a landmark-based navigation approach. It analyzes initial clues to determine potential landmarks and uses vision-language models to identify and navigate towards these landmarks until it finds the injured person and eventually the stretcher/ambulance.


## Discussion

### Landmark-Based Navigation
The agent parses the initial text clue to identify potential landmarks that might lead to the injured person. 
For example, the clue is "The injured person is by the roadside garbage bin, and there is a car nearby."
then landmarks can be [roadside, car, garbage bin, injured person]. The agent will detect them in priority order and move towards them, ultimately finding the injured person.


### Action and Observation Buffers
- **Action Buffer**: Maintains a queue of pending actions to be executed, allowing the agent to plan multiple steps ahead.
- **Observation Buffer**: Coordinates with the action buffer to determine which observations should be saved for later analysis.
These two buffers allow the VLM to perform continuous action control on the agent during the loop process.


### Some Details

1. **Obstacle Avoidance**: The agent alternates between walking and jumping to navigate around obstacles when searching for the injured person.

2. **Repeated Pickup Attempts**: After finding the target person, the agent attempts to pick them up after each move to compensate for the limited precision of VLM-based control.

3. **Optimal Placement Distance**: When placing the injured person on the stretcher, the agent backs up two steps before dropping, which has proven to be an appropriate distance.

4. **Multi-Directional Observation**: The agent captures images from multiple directions and concatenates them to provide the VLM with a comprehensive view of the surroundings.

5. **Visual Guidance Lines**: When moving forward, the agent divides the image into left, middle, and right regions using vertical lines to help the VLM better control direction.

## Limitations

1. **Limited Use of Visual Clues**: The agent primarily relies on textual clues and doesn't fully utilize visual information that might be present in the environment.

2. **Poor Performance on Long-Distance Tasks**: The agent struggles with missions requiring long-distance travel because:
   - Landmarks cannot be effectively deployed as a VLM-detectable continuous route
   - The current structure involves many small movement steps rather than efficient long strides

3. **Primitive Obstacle Avoidance**: When blocked, the agent uses a simple random movement strategy (left-forward or right-forward) rather than intelligent path planning.

4. **VLM Hallucination Issues**: The agent occasionally suffers from VLM hallucinations, particularly with smaller models, which can lead to navigation errors and inefficiencies.

## Code Structure and Flow

### Main Components

1. **Initialization** (`__init__`): Sets up the agent state, buffers, and tracking variables.

2. **Prediction Loop** (`predict`): The main function called by the environment that:
   - Processes current observations
   - Selects the next action based on the current phase
   - Manages the transition between different phases of the rescue mission

3. **Phase Handlers**:
   - `_handle_initial_phase`: Analyzes the clue and initializes search
   - `_handle_search_phase`: Searches for and moves toward the injured person
   - `_handle_rescue_phase`: Picks up the injured person
   - `_handle_return_phase`: Searches for the stretcher/ambulance
   - `_handle_placement_phase`: Places the person on the stretcher

4. **Action Processors**:
   - `_start_search`: Initiates a search by rotating and observing in different directions
   - `_start_move_to_landmark`: Begins movement toward an identified landmark
   - `_process_search_result`: Analyzes collected observations to find landmarks
   - `_process_move_result`: Determines if movement toward a landmark was successful

5. **Utility Functions**:
   - `analyse_initial_clue`: Parses the initial textual clue to extract directions and landmarks
   - `encode_image_array`: Converts image arrays to base64 for API calls
   - `concatenate_images`: Combines multiple observation images
   - `add_vertical_lines`: Adds guidance lines to images for movement control

### Execution Flow

1. The agent starts by analyzing the initial clue to identify potential landmarks and direction.
2. It enters the search phase, rotating to observe the surroundings and using VLM to identify landmarks.
3. Upon finding a landmark, it moves toward it while continuously checking if the injured person is visible.
4. After finding and picking up the injured person, it transitions to the return phase.
5. In the return phase, it searches for the stretcher/ambulance.
6. Finally, in the placement phase, it approaches the stretcher and places the injured person on it.

Throughout this process, the agent uses the action and observation buffers to queue up sequences of actions, making the control more efficient and reducing the number of API calls needed.

## Potential Improvements

1. Incorporate visual clue analysis to supplement textual landmark information
2. Implement more sophisticated path planning for long-distance navigation
3. Develop better obstacle avoidance strategies using reinforcement learning
4. Add confidence scoring to VLM outputs to reduce the impact of hallucinations
5. Optimize the movement patterns to reduce the number of steps needed for navigation
