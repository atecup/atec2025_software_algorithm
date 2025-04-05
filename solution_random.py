import random

class AlgSolution: # 类名不能变

    def __init__(self):
        pass

    def reset(self, reference_text, reference_image):
        # This function is called before each new test data inference;
        # reference_text and reference_image are the reference text and reference image for the data, respectively. If None, it means there is none.
        # reference_image is a base64 encoded png image
        # This function completes some state initialization, cache clearing, etc.
        pass

    def predicts(self, ob, success):
        # ob: a base64 encoded png image with a resolution of 320*320, representing the current observation image
        # success: a boolean variable, True/False represents whether the last command was successfully executed
        # Most of the time, success is True. When the player performs the "carry" operation to lift the wounded, if it is not within the allowable distance, the action may fail.

        linear = random.randint(-100, 100)
        angular = random.randint(-30,30)

        action = {
            'angular': angular,  # [-30, 30]
            'velocity': linear,  # [-100, 100],
            'viewport': 0,  # {0: stay, 1: look up, 2: look down},
            'interaction': 0,  # {0: stand, 1: jump, 2: crouch, 3: carry, 4: drop, 5: open door}
        }
        return action