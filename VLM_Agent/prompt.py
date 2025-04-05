def initial_clue_prompt(clue):
    p = f"""
        You are a rescue robot, and now you have the following initial prompts about the rescue target:
        {clue}

        Please analyze two major questions:
        1. Does the prompt mention where the first target is located on the robot? (Front, right side, etc.)
        2. What objects or symbols are mentioned in the prompt? According to the logical description of the prompt, arrange the mentioned landmark objects as nodes in the order from the starting point (away from the injured person) to the ending point (injured person).
        
        Output format with XML tags:
        <a> Your step-by-step think process </a>
        <b>first target's direction (front/back/left/right/none)</b>
        <c>Sorted landmark objects<c>

        Example:
        Tip: The injured person is under a fruit tree. After the fruit tree enters the orchard, turn left. There is a fence outside the orchard and a car is parked.
        <a> (think process) </a>
        <b>front</b>
        <c>car,fence,orchard entrance,fruit tree,injured person</c>


    """
    return p




def search_prompt(landmark_list):

    landmark_str = ", ".join(landmark_list)

    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image is a concatenation of three images, reflecting the robot's field of view on the left, front, and right sides, respectively.

        You now have a list of landmarks:[{landmark_str}]
        the list was sort in descending order of priority.

        Please complete the following tasks:
        1. Please analyze the information and objects contained in each image separately;
        2. Check if the objects in the list appear within the field of view.
        

        Use XML tags to output results in the following format:
        <a>Check whether the objects in the list appear in order based on the information within the field of view</a>
        <b>the object in the views with the highest priority, or outputs NONE to indicate that all objects in the list are invisible</b> (When describing an object, please ensure consistency with the expression in the list)
        <c> choose one from left/front/right (Which image does the chosen object belong to) </c>

        Example:
        <a>The left view includes red car, trash can, and lawn. There is a wall in front, which should be the wall of the house. There is a swimming pool on the right, with parasols and lounge chairs next to it. In order, first check the injured person - none of the three pictures are present. Then there is the red car, which exists in the field of view and belongs to the left view. I should output red car.</a>
        <b>red car</b>
        <c>left</c>

    """
    return p





def search_prompt_begin(landmark_list):

    landmark_str = ", ".join(landmark_list)

    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image now displays the visual field in front of the robot.

        You now have a list of landmarks:[{landmark_str}]
        the objects in the list are some iconic objects that may be seen on the path from your current location to the injured person. The objects at the front are of low priority, while the injured at the bottom have the highest priority.

        Please complete the following tasks:
        1. Analyze which objects are present in the field of view;
        2. Corresponding to the iconic objects in the list, find the object with the highest priority in the field of view.
        Please note that the description should be consistent with the description in the list.

        Use XML tags to output results in the following format:
        <a>Analysis of information within the field of view</a>
        <b>The highest priority landmark object (If there is no landmark object in the list within the field of view, output NONE)</b>

        Example:
        <a>In the field of view, you can see the entrance of the orchard, the fence, and the garbage bin, both of which are landmarks on the list. Since the priority of the entrance is higher than that of the fence, I will choose the fence output.</b>
        <b>fence</b>

    """
    return p



def search_prompt_back(landmark_list):

    landmark_str = ", ".join(landmark_list)

    p = f"""

        You are a rescue robot, you need to find the wounded.

        The input RGB image is a concatenation of four images, reflecting the robot's field of view on the front, right ,back and left sides, respectively.
        
        You need to find the stretcher within your field of view (if it exists). Stretchers will always appear next to the ambulance, so when you cannot see the stretcher, you should output the ambulance.

        Please complete the following tasks:

        1. Please analyze the information and objects contained in each image separately;
        2. Check if there is an ambulance or stretcher in the field of view. If there is, output it. When both are visible, prioritize outputting the stretcher and only output one object.

        Use XML tags to output results in the following format:

        <a>Check if there is a stretcher or ambulance in the field of view</a>
        <b>stretcher/ambulance/NONE</b>
        <c> select one from front/right/back/left (which image does the selected object belong to)</c>

        example:
        <a>The left view includes red cars, trash cans, and lawns. There are stretchers and ambulances ahead. There is a swimming pool on the right, with parasols and lounge chairs next to it. The image behind is a wall. The stretcher and ambulance are both within sight, I should prioritize the delivery of the stretcher.</a>
        <b>stretcher</b>
        <c>front</c>

    """
    return p


def move_forward_prompt(target):
    p = f"""

You are a rescue robot, you need to find the wounded.

The input RGB image now displays the visual field in front of the robot. You are currently moving towards the target :[{target}].
The image is divided into three parts by two red vertical lines: left, middle, right. Please identify the following issue:
1. Is the target still within sight?
2. Is the target mainly located on the left, middle, or right side of the image?

Use XML tags to output results in the following format:
<a>yes/no (Determine if the target is still within the field of view)</a>
<b>left/middle/right(Determine which area the target is in)</b> 

Example:
<a>yes</a>
<b>middle</b> 

    """
    return p








def search_move_prompt():
    p = f"""

        You are a rescue robot, you need to explore the environment and find the wounded.
        The input RGB image now is a concatenation of three images, showing the visual content of the robot's left, front, and right sides respectively.
        Please analyze the field of view information within each of the three images and choose the direction that is more worth exploring.
        How to determine if it is worth exploring:
        If there are distant extensions or new areas such as corners or channels connecting in the picture, it is more worth exploring; If it's a wall or a corner, then it's obviously not worth exploring.
        Use XML tags to output results in the following format:
        <think> Your analysis process </think>
        <output>The direction you choose can be left, front, or right</output>

        Example:
        <think> In front of me is a wall, and on the right is an empty space, which seems to have no exploratory value. There is a door on my left, and the space inside the door is an unknown area worth exploring. </think>
        <output>left</output>
    """
    return p


def access_search_prompt():
    p = f"""
        # background

        You are a rescue robot that needs to explore unknown environments and find mission targets (wounded and stretchers)
        A person was injured and lying next to the red car on the right side of the house.

        # Input information:
        RGB image: composed of four images pieced together, representing the content of your field of view in the front, right, back, and left directions.

        Please perform the following tasks in order:

        1. Analyze the information within your field of view to determine if you are currently on the road?
        2. If you are not on the road, please find the nearest intersection to help you move from your current area to the road.

        Output (including XML format):
        <a>whether you are currently on the road</a> (select from yes, no)
        <b>To move the target, you need to describe the intersection you want to pass through</b>
        <c> In which direction is the intersection within the field of view? (Select from front, right, back, left)</c>


        Example:
        <a>no</a>
        <b>the left side of the white fence can be accessed</b>
        <c>front</c>


    """

    return p 





def move_obstacle_prompt():
    p = f"""
        You are helping a robot move, and the incoming RGB image is its downward view. 
        Please confirm if there are any horizontal obstacles in the view that may hinder our movement, such as a wall, a horizontal fence, etc. And other objects, such as a tree or a box, although they are also obstacles, their width is limited, so we can easily navigate around them, and these are not considered horizontal obstacles.
        In addition, obstacles also need to have sufficient height, and if they are just objects on the ground, they can also be ignored.

        Please output according to the format, including XML tags:
        <a>Whether there are horizontal obstacles</a> (Choose from 0 and 1, where 0 represents none and 1 represents presence)
        <b>If it exists, describe what obstacle it is</b>


        Example:
        <a>1</a>
        <b>There is a white fence in front of me</b>

        """
    return p