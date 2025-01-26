import numpy as np
import settings
from random import uniform, randrange, choice
import pygame
import copy
import random

from EpsilonScheduler import EpsilonScheduler
from springer import Springer

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

class SpringerLogic_QLearning_v2:
    def __init__(self, leg_angle_range, leg_angle_buckets, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize the Q-learning logic for Springer objects.
        """
        self.run_animation = False

        # self.fall_penalty_duration_fraction = 0.05
        self.fall_penalty_frames = settings.rewards["all_penalty_frames"]
        self.speed_average_duration = settings.rewards["speed_average_duration"]
        self.reward_mulitplier = settings.rewards["reward_mulitplier"]
        self.penalty_multiplier = settings.rewards["penalty_multiplier"]

        self.leg_angle_range = leg_angle_range  # Tuple (min_angle, max_angle)
        self.leg_angle_buckets = leg_angle_buckets
        self.action_number = len(Springer.ACTIONS)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.knowledge = {}  # Q-table, represented as a dictionary
        for i in range(leg_angle_buckets):
            for j in range(3): # speed buckets
                for k in range(2): # jump
                    for l in range(2):
                        self.knowledge[(i, j, k, l)] = np.ones(len(Springer.ACTIONS)) # Assures the dictionary will be in readable order
                        if k == 0:
                            self.knowledge[(i, j, k, l)][0] = 0
                        if l == 0:
                            self.knowledge[(i, j, k, l)][1] = 0
                            self.knowledge[(i, j, k, l)][2] = 0

        self.iteration_history = []
        self.last_score = None
        self.total_rewards = 0

    def quantize_leg_angle(self, leg_angle: float) -> int:
        """
        Quantize leg angle into discrete buckets within the specified range.
        """
        min_angle, max_angle = self.leg_angle_range
        if leg_angle < min_angle or leg_angle > max_angle:
            raise ValueError(f"Leg angle {leg_angle} out of range [{min_angle}, {max_angle}]")
        
        bucket_size = (max_angle - min_angle) / self.leg_angle_buckets
        return int((leg_angle - min_angle) // bucket_size)

    def quantize_speed(self, x_speed: float) -> int:
        """
        Quantize height into discrete buckets within the specified range.
        """
        if x_speed < -0.3:
            return 0
        if x_speed > 0.3:
            return 2
        return 1

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
        speed = self.quantize_speed(self.retrieve_from_state(state, "x_speed"))
        can_jump = state["can_jump"]
        can_shift = state["can_shift"]
        state_key = (leg_angle, speed, can_jump, can_shift)

        # Initialize Q-values for this state if not already present
        if state_key not in self.knowledge:
            self.knowledge[state_key] = np.ones(self.action_number)

        if isinstance(self.epsilon, EpsilonScheduler):
            epsilon_value = self.epsilon.get_epsilon()
        else:
            epsilon_value = self.epsilon

        if np.random.uniform(0, 1) < epsilon_value:
            # Explore: choose random action
            return np.random.choice(Springer.ACTIONS)
        else:
            # Exploit: choose best action
            # print(f"Choosing action for state: {state_key}, {self.knowledge[state_key]}")
            # print(self.knowledge[state_key])

            # action_index = np.argmax(self.knowledge[state_key]) # TODO: THE OLD WAY

            state_knowledge = copy.deepcopy(self.knowledge[state_key])
            state_knowledge -= state_knowledge.min()
            if sum(state_knowledge) == 0:
                return Springer.ACTIONS[random.randrange(3)]
            action_index = random.choices(range(len(state_knowledge)), weights=state_knowledge, k=1)[0]

            # print("FROM", state_knowledge, "CHOSEN", ["jump", "right", "left"][action_index])

            # print(f"Action index: {action_index}, {self.knowledge[state_key][action_index]}")
            return Springer.ACTIONS[action_index]

    def update_knowledge(self, state: dict, action: str):
        """
        Update the Q-table using the Q-learning update rule.
        """
        state["action"] = action
        self.iteration_history.append(state)
        self.last_score = state["x_distance"]
        if state["marked_for_removal"]:
            # penalty_duration = int(max(0, len(self.iteration_history) * self.fall_penalty_duration_fraction - 1))
            penalty_duration = 0 if self.fall_penalty_frames > len(self.iteration_history) else self.fall_penalty_frames
            # REWARDS
            for state_index, h1_state in enumerate(self.iteration_history[:-penalty_duration]):
                # print("Rewarding action", h1_state["action"])
                h2_state = self.iteration_history[state_index+1]

                # Calculate reward
                speed_end_index = min(penalty_duration-1, state_index+self.speed_average_duration)
                future_position = self.iteration_history[speed_end_index]["x_distance"]
                if state_index == speed_end_index:
                    reward = 0
                else:
                    reward = self.reward_mulitplier * (future_position - h1_state["x_distance"]) / (speed_end_index - state_index)
                # reward += h1_state["x_speed"]

                leg_angle = self.quantize_leg_angle(self.retrieve_from_state(h1_state, "leg_angle"))
                speed = self.quantize_speed(self.retrieve_from_state(h1_state, "x_speed"))
                can_jump = h1_state["can_jump"]
                can_shift = h1_state["can_shift"]
                state_key = (leg_angle, speed, can_jump, can_shift)

                next_leg_angle = self.quantize_leg_angle(self.retrieve_from_state(h2_state, "leg_angle"))
                next_speed = self.quantize_speed(self.retrieve_from_state(h2_state, "x_speed"))
                next_can_jump = h2_state["can_jump"]
                next_can_shift = h2_state["can_shift"]
                next_state_key = (next_leg_angle, next_speed, next_can_jump, next_can_shift)

                # print(f"Updating knowledge for state: {state_key}, {action}, {reward}, {next_state_key}")

                # Initialize Q-values for current and next states if not present
                if state_key not in self.knowledge:
                    self.knowledge[state_key] = np.ones(self.action_number)
                if next_state_key not in self.knowledge:
                    self.knowledge[next_state_key] = np.ones(self.action_number)

                action_index = Springer.ACTIONS.index(h1_state["action"])
                best_next_action = np.max(self.knowledge[next_state_key])
                old_value = self.knowledge[state_key][action_index]
                
                # reward = -reward
                self.total_rewards += reward
                self.knowledge[state_key][action_index] = (1 - self.learning_rate) * old_value + self.learning_rate * reward
                # self.knowledge[state_key][action_index] = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * best_next_action)
                # if reward > 0:
                #     self.knowledge[state_key][action_index] = (1 - self.learning_rate) * old_value + self.learning_rate * reward

                if self.knowledge[state_key][action_index] < 0:
                    self.knowledge[state_key][action_index] = 0.01

                if can_jump == 0:
                    self.knowledge[state_key][Springer.ACTIONS.index("jump")] = 0
                if can_shift == 0:
                    self.knowledge[state_key][Springer.ACTIONS.index("left")] = 0
                    self.knowledge[state_key][Springer.ACTIONS.index("right")] = 0

                if self.run_animation:
                    print("Rewarding", state_key, "for", h1_state["action"], "with", self.learning_rate * reward, "iteration", state_index)
                # print(self.knowledge)
                # print("STATE: ", h1_state)
            # PENALTIES
            if not state["dont_apply_penalty"]:
                for state_index, h1_state in enumerate(self.iteration_history[-penalty_duration:-1]):
                    leg_angle = self.quantize_leg_angle(self.retrieve_from_state(h1_state, "leg_angle"))
                    speed = self.quantize_speed(self.retrieve_from_state(h1_state, "speed"))
                    can_jump = h1_state["can_jump"]
                    can_shift = h1_state["can_shift"]
                    state_key = (leg_angle, speed, can_jump, can_shift)
                    # if h1_state["action"] is None:
                    action_index = Springer.ACTIONS.index(h1_state["action"])
                    penalty = (len(self.iteration_history) - state_index) * self.penalty_multiplier
                    self.total_rewards -= penalty
                    self.knowledge[state_key][action_index] -= penalty
                    if self.knowledge[state_key][action_index] < 0:
                        self.knowledge[state_key][action_index] = 0.01
                    if self.run_animation:
                        print("Penalizing", state_key, "for", h1_state["action"], "with", penalty, "iteration", state_index+penalty_duration)

                        
                    if can_jump == 0:
                        self.knowledge[state_key][Springer.ACTIONS.index("jump")] = 0
                    if can_shift == 0:
                        self.knowledge[state_key][Springer.ACTIONS.index("left")] = 0
                        self.knowledge[state_key][Springer.ACTIONS.index("right")] = 0
                # print("STATE: ", h1_state)


class SpringerLogic_Manual:
    def __init__(self):
        # A - left, W - jump, D - right

        # The variables below are here for compatibility with QLearning but they don't affect anything
        self.height_range = 0
        self.leg_angle_range = 0
        self.height_buckets = 0
        self.leg_angle_buckets = 0
        self.action_number = len(Springer.ACTIONS)
        self.learning_rate = 0
        self.discount_factor = 0
        self.epsilon = 0
        self.knowledge = {}
        self.last_score = 0
        self.total_rewards = 0

    def chooseAction(self, state: dict) -> str:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            return "jump"
        if keys[pygame.K_a]:
            return "left"
        if keys[pygame.K_d]:
            return "right"
        # return ""
    
    def quantize_leg_angle(self, leg_angle: float) -> int:
        pass

    def quantize_height(self, height: float) -> int:
        pass

    def retrieve_from_state(self, state: dict, parameter: str):
        pass

    def update_knowledge(self, x, y):
        pass

    def apply_reward(self, state: dict) -> float:
        pass

class SpringerLogic_QLearning:
    def __init__(self, height_range, leg_angle_range, height_buckets, leg_angle_buckets, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize the Q-learning logic for Springer objects.
        """
        self.height_range = height_range  # Tuple (min_height, max_height)
        self.leg_angle_range = leg_angle_range  # Tuple (min_angle, max_angle)
        self.height_buckets = height_buckets
        self.leg_angle_buckets = leg_angle_buckets
        self.action_number = len(Springer.ACTIONS)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.knowledge = {}  # Q-table, represented as a dictionary
        for i in range(leg_angle_buckets):
            for j in range(height_buckets):
                self.knowledge[(i, j)] = np.zeros((4)) # Assures the dictionary will be in readable order

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
        if height < min_height:
            height = min_height
        if height > max_height:
            height = max_height
        
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
            return np.random.choice(Springer.ACTIONS)
        else:
            # Exploit: choose best action
            # print(f"Choosing action for state: {state_key}, {self.knowledge[state_key]}")
            action_index = np.argmax(self.knowledge[state_key])
            # print(f"Action index: {action_index}, {self.knowledge[state_key][action_index]}")
            return ["jump", "right", "left"][action_index]

    def update_knowledge(self, state: dict, action: str, reward: float, next_state: dict):
        """
        Update the Q-table using the Q-learning update rule.
        """

        if state is next_state:
            raise Exception("Error: State and next state are the same object")
        leg_angle = self.quantize_leg_angle(self.retrieve_from_state(state, "leg_angle"))
        height = self.quantize_height(self.retrieve_from_state(state, "height"))
        state_key = (leg_angle, height)

        next_leg_angle = self.quantize_leg_angle(self.retrieve_from_state(next_state, "leg_angle"))
        next_height = self.quantize_height(self.retrieve_from_state(next_state, "height"))
        next_state_key = (next_leg_angle, next_height)

        # print(f"Updating knowledge for state: {state_key}, {action}, {reward}, {next_state_key}")

        # Initialize Q-values for current and next states if not present
        if state_key not in self.knowledge:
            self.knowledge[state_key] = np.zeros(self.action_number)
        if next_state_key not in self.knowledge:
            self.knowledge[next_state_key] = np.zeros(self.action_number)

        action_index = Springer.ACTIONS.index(action)
        best_next_action = np.max(self.knowledge[next_state_key])
        old_value = self.knowledge[state_key][action_index]

        self.knowledge[state_key][action_index] = (1- self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * best_next_action)

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


class SpringerLogic_QLearning_can_jump:
    def __init__(self, leg_angle_range, leg_angle_buckets, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize the Q-learning logic for Springer objects.
        """
        self.leg_angle_range = leg_angle_range  # Tuple (min_angle, max_angle)
        self.leg_angle_buckets = leg_angle_buckets
        self.action_number = len(Springer.ACTIONS)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.knowledge = {}  # Q-table, represented as a dictionary
        for i in range(leg_angle_buckets):
            for j in [0, 1]:  # Binary values for "can_jump" state
                self.knowledge[(i, j)] = np.zeros((self.action_number))  # Assures the dictionary will be in readable order

    def quantize_leg_angle(self, leg_angle: float) -> int:
        """
        Quantize leg angle into discrete buckets within the specified range.
        """
        min_angle, max_angle = self.leg_angle_range
        if leg_angle < min_angle or leg_angle > max_angle:
            raise ValueError(f"Leg angle {leg_angle} out of range [{min_angle}, {max_angle}]")
        
        bucket_size = (max_angle - min_angle) / self.leg_angle_buckets
        return int((leg_angle - min_angle) // bucket_size)

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
        can_jump = self.retrieve_from_state(state, "can_jump")  # Use the new "can_jump" state key
        state_key = (leg_angle, can_jump)

        # Initialize Q-values for this state if not already present
        if state_key not in self.knowledge:
            self.knowledge[state_key] = np.zeros(self.action_number)

        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: choose random action
            return np.random.choice(Springer.ACTIONS)
        else:
            # Exploit: choose best action
            action_index = np.argmax(self.knowledge[state_key])
            return Springer.ACTIONS[action_index]

    def update_knowledge(self, state: dict, action: str, reward: float, next_state: dict):
        """
        Update the Q-table using the Q-learning update rule.
        """
        if state is next_state:
            raise Exception("Error: State and next state are the same object")
        leg_angle = self.quantize_leg_angle(self.retrieve_from_state(state, "leg_angle"))
        can_jump = self.retrieve_from_state(state, "can_jump")
        state_key = (leg_angle, can_jump)

        next_leg_angle = self.quantize_leg_angle(self.retrieve_from_state(next_state, "leg_angle"))
        next_can_jump = self.retrieve_from_state(next_state, "can_jump")
        next_state_key = (next_leg_angle, next_can_jump)

        # Initialize Q-values for current and next states if not present
        if state_key not in self.knowledge:
            self.knowledge[state_key] = np.zeros(self.action_number)
        if next_state_key not in self.knowledge:
            self.knowledge[next_state_key] = np.zeros(self.action_number)

        action_index = Springer.ACTIONS.index(action)
        best_next_action = np.max(self.knowledge[next_state_key])
        old_value = self.knowledge[state_key][action_index]

        self.knowledge[state_key][action_index] = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * best_next_action)

    def apply_reward(self, state: dict) -> float:
        """
        Apply reward to the agent based on the current state.
        """
        return self.retrieve_from_state(state, "x_distance")
