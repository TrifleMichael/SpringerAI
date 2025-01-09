import numpy as np
import settings
from random import uniform, randrange, choice

class SpringerLogic:
    def retrieveFromState(self, state: dict, parameter: str) -> str:
        if parameter not in state:
            raise Exception(f"Error: Incorrectly accessign state, parameter {parameter} does not exist in {state}")
        return state[parameter]
        
    def chooseAction(self, state: dict) -> str:
        step = self.retrieveFromState(state, "step")

        # Randomized timestamps for action
        jump_step = randrange(190, 210)
        shift_step = randrange(175, 185)

        if step == jump_step:
            return "jump"
        elif step == shift_step:
            if uniform(0, 1) < 0.5:
                return "right"
            else:
                return "left"
        else:
            ""
            
        if self.retrieveFromState(state, "marked_for_removal"):
            print("Managed to score:", self.retrieveFromState(state, "x_distance"))

class SpringerLogic_QLearning:
    def __init__(self, height_range, leg_angle_range, height_buckets, leg_angle_buckets, 
                 action_number=4, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize the Q-learning logic for Springer objects.
        """
        self.height_range = height_range  # Tuple (min_height, max_height)
        self.leg_angle_range = leg_angle_range  # Tuple (min_angle, max_angle)
        self.height_buckets = height_buckets
        self.leg_angle_buckets = leg_angle_buckets
        self.action_number = action_number  # Includes "jump", "right", "left", and "" (do nothing)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.knowledge = {}  # Q-table, represented as a dictionary

    def quantize_leg_angle(self, leg_angle: float) -> int:
        """
        Quantize leg angle into discrete buckets within the specified range.
        """
        min_angle, max_angle = self.leg_angle_range
        if leg_angle < min_angle or leg_angle > max_angle:
            raise ValueError(f"Leg angle {leg_angle} out of range [{min_angle}, {max_angle}]")
        
        bucket_size = (max_angle - min_angle) / self.leg_angle_buckets
        return int((leg_angle - min_angle) // bucket_size)

    def quantize_height(self, height: float) -> int:
        """
        Quantize height into discrete buckets within the specified range.
        """
        min_height, max_height = self.height_range
        if height < min_height or height > max_height:
            raise ValueError(f"Height {height} out of range [{min_height}, {max_height}]")
        
        bucket_size = (max_height - min_height) / self.height_buckets
        return int((height - min_height) // bucket_size)

    def retrieve_from_state(self, state: dict, parameter: str):
        """
        Retrieve a value from the state dictionary with error checking.
        """
        if parameter not in state:
            raise Exception(f"Error: Incorrectly accessing state, parameter {parameter} does not exist in {state}")
        return state[parameter]

    def chooseAction(self, state: dict) -> str:
        """
        Choose an action using epsilon-greedy policy.
        """
        leg_angle = self.quantize_leg_angle(self.retrieve_from_state(state, "leg_angle"))
        height = self.quantize_height(self.retrieve_from_state(state, "height"))
        state_key = (leg_angle, height)

        # Initialize Q-values for this state if not already present
        if state_key not in self.knowledge:
            self.knowledge[state_key] = np.zeros(self.action_number)

        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: choose random action
            return np.random.choice(["jump", "right", "left", ""])
        else:
            # Exploit: choose best action
            # print(f"Choosing action for state: {state_key}, {self.knowledge[state_key]}")
            action_index = np.argmax(self.knowledge[state_key])
            # print(f"Action index: {action_index}, {self.knowledge[state_key][action_index]}")
            return ["jump", "right", "left", ""][action_index]

    def update_knowledge(self, state: dict, action: str, reward: float, next_state: dict):
        """
        Update the Q-table using the Q-learning update rule.
        """
        leg_angle = self.quantize_leg_angle(self.retrieve_from_state(state, "leg_angle"))
        height = self.quantize_height(self.retrieve_from_state(state, "height"))
        state_key = (leg_angle, height)

        next_leg_angle = self.quantize_leg_angle(self.retrieve_from_state(next_state, "leg_angle"))
        next_height = self.quantize_height(self.retrieve_from_state(next_state, "height"))
        next_state_key = (next_leg_angle, next_height)

        # Initialize Q-values for current and next states if not present
        if state_key not in self.knowledge:
            self.knowledge[state_key] = np.zeros(self.action_number)
        if next_state_key not in self.knowledge:
            self.knowledge[next_state_key] = np.zeros(self.action_number)

        action_index = ["jump", "right", "left", ""].index(action)
        best_next_action = np.max(self.knowledge[next_state_key])
        old_value = self.knowledge[state_key][action_index]

        # Q-learning formula
        # self.knowledge[state_key][action_index] += self.learning_rate * (
        #     reward + self.discount_factor * best_next_action - self.knowledge[state_key][action_index]
        # )
        self.knowledge[state_key][action_index] = (1- self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * best_next_action)
        if reward < 0:
            print(f"Penalized action: {state_key}, {action}, {reward}, {next_state_key}\nv: {old_value} -> {self.knowledge[state_key][action_index]}")

    def apply_reward(self, state: dict) -> float:
        """
        Apply reward to the agent based on the current state.
        """
        return self.retrieve_from_state(state, "x_distance")



class SpringerLogic_simple:
    def retrieveFromState(self, state: dict, parameter: str) -> str:
        if parameter not in state:
            raise Exception(f"Error: Incorrectly accessign state, parameter {parameter} does not exist in {state}")
        return state[parameter]
        
    def chooseAction(self, state: dict) -> str:
        step = self.retrieveFromState(state, "step")
        if step == 200:
            return "jump"
        elif step == 180:
            return "right"
        else:
            ""

