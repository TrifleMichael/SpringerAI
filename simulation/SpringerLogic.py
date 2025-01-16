import numpy as np
import settings
from random import uniform, randrange, choice
import random

import tensorflow as tf
import numpy as np

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
            return ["jump", "right", "left", ""][action_index]

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



# class SpringerLogic_DQL_can_jump:
#     def __init__(self, leg_angle_range, learning_rate=0.001, discount_factor=0.9, epsilon=0.1):
#         """
#         Initialize the Deep Q-learning logic for Springer objects.
#         """
#         self.leg_angle_range = leg_angle_range  # Tuple (min_angle, max_angle)
#         self.relevant_fields = ["leg_angle", "x_speed", "x_leg_distance", "can_jump"]
#         self.state_dim = len(self.relevant_fields)  # Based on filtered state fields
#         self.action_dim = len(Springer.ACTIONS)
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.epsilon = epsilon

#         # Create the Q-network
#         self.model = self.build_model()
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

#         self.min_max = {"min": [0, -6.316734217467598, -352.3168275864597, 0], "max": [3.1415887363412036, 8.831108269139975, 360.93097140912494, 1]}

#     def build_model(self):
#         """Build a neural network model for approximating Q-values."""
#         model = tf.keras.Sequential([
#             tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
#             tf.keras.layers.Dense(32, activation='relu'),
#             tf.keras.layers.Dense(self.action_dim)
#         ])
#         return model

#     def filter_and_transform_state(self, state: dict) -> tf.Tensor:
#         """
#         Filter relevant fields from the state dictionary and transform them into a TensorFlow tensor.
#         """
#         # filtered_state = [state[field] for field in self.relevant_fields if field in state]

#         filtered_state = []
#         for i, field in enumerate(self.relevant_fields):
#             if field in state:
#                 value = state[field]
#                 # Clamp value to min and max
#                 min_val = self.min_max["min"][i]
#                 max_val = self.min_max["max"][i]
#                 value = max(min_val, min(value, max_val))
#                 # Normalize value
#                 normalized_value = (value - min_val) / (max_val - min_val)
#                 filtered_state.append(normalized_value)
#             else:
#                 raise KeyError(f"State is missing required field: {field}")
#         return tf.convert_to_tensor(filtered_state, dtype=tf.float32)

#     def chooseAction(self, state: dict) -> int:
#         """
#         Choose an action using epsilon-greedy policy.
#         """
#         filtered_state = self.filter_and_transform_state(state)
#         if np.random.uniform(0, 1) < self.epsilon:
#             # Explore: choose random action
#             return np.random.randint(0, self.action_dim)
#         else:
#             # Exploit: choose the best action
#             filtered_state = tf.expand_dims(filtered_state, axis=0)  # Add batch dimension
#             q_values = self.model(filtered_state, training=False)
#             return tf.argmax(q_values[0]).numpy()

#     @tf.function(reduce_retracing=True)
#     def _update_knowledge_internal(self, filtered_state: tf.Tensor, action_index: int, reward: tf.Tensor, filtered_next_state: tf.Tensor, done: bool):
#         """
#         Internal method for updating the Q-network using TensorFlow tensors.
#         """
#         # Compute the target Q-value
#         next_q_values = self.model(filtered_next_state, training=False)
#         target_q_value = reward
#         if not done:
#             target_q_value += tf.cast(self.discount_factor, tf.float32) * tf.reduce_max(next_q_values[0])

#         # Compute the current Q-value
#         with tf.GradientTape() as tape:
#             q_values = self.model(filtered_state, training=True)
#             q_value = q_values[0, action_index]  # Index directly for consistent shape

#             # Compute the loss
#             loss = tf.keras.losses.MSE([target_q_value], [q_value])

#         # Update the model weights
#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

#     def update_knowledge(self, state: dict, action: str, reward: float, next_state: dict, done: bool):
#         """
#         Public method to update the Q-network. Converts inputs to TensorFlow tensors and calls the internal method.
#         """
#         reward = tf.convert_to_tensor(reward, dtype=tf.float32)  # Ensure reward is a float32 tensor
#         filtered_state = tf.expand_dims(self.filter_and_transform_state(state), axis=0)  # Add batch dimension
#         filtered_next_state = tf.expand_dims(self.filter_and_transform_state(next_state), axis=0)  # Add batch dimension

#         # Map the action string to its index
#         action_index = Springer.ACTIONS.index(action)

#         # Call the internal method
#         self._update_knowledge_internal(filtered_state, action_index, reward, filtered_next_state, done)



#     def save_model(self, path):
#         """Save the model to the specified path."""
#         self.model.save(path)

#     def load_model(self, path):
#         """Load the model from the specified path."""
#         self.model = tf.keras.models.load_model(path)

class EpsilonScheduler:
    def __init__(self, start: float, end: float, decay_rate: float):
        """
        Initialize an epsilon scheduler for epsilon-greedy strategy.

        :param start: Initial epsilon value.
        :param end: Final epsilon value.
        :param decay_rate: The rate at which epsilon decays.
        """
        self.epsilon = start
        self.start = start
        self.end = end
        self.decay_rate = decay_rate

    def get_epsilon(self) -> float:
        """Return the current epsilon value."""
        return self.epsilon

    def decay(self):
        """Decay the epsilon value based on the decay rate."""
        self.epsilon = max(self.end, self.epsilon * self.decay_rate)

    def __str__(self):
        return f"EpsilonScheduler(start={self.start}, end={self.end}, decay_rate={self.decay_rate}) current={self.epsilon}"    


class SpringerLogic_DQL_can_jump:
    def __init__(self, leg_angle_range, learning_rate=0.001, discount_factor=0.9, epsilon=0.1, buffer_capacity=10000, batch_size=64):
        """
        Initialize the Deep Q-learning logic for Springer objects.
        """
        self.leg_angle_range = leg_angle_range  # Tuple (min_angle, max_angle)
        self.relevant_fields = ["leg_angle", "x_speed", "x_leg_distance", "can_jump"]
        self.state_dim = len(self.relevant_fields)  # Based on filtered state fields
        self.action_dim = len(Springer.ACTIONS)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size

        # Create the Q-network and target network
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()  # Ensure target model starts with the same weights
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_capacity = buffer_capacity

        # Store min and max values for normalization
        self.min_max = {"min": [0, -10.0, -352.3168275864597, 0], "max": [3.1415887363412036, 10, 360.93097140912494, 1]}


    def build_model(self):
        """Build a neural network model for approximating Q-values."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_dim)
        ])
        return model

    def update_target_network(self):
        """Update the target network to match the main network."""
        self.target_model.set_weights(self.model.get_weights())

    def filter_and_transform_state(self, state: dict) -> tf.Tensor:
        """
        Filter relevant fields from the state dictionary and transform them into a TensorFlow tensor.
        Additionally, normalize the values based on predefined min and max ranges with clamping.
        """
        relevant_fields = ["x_distance", "leg_angle", "speed", "x_speed"]
        filtered_state = []
        for i, field in enumerate(relevant_fields):
            if field in state:
                value = state[field]
                # Clamp value to min and max
                min_val = self.min_max["min"][i]
                max_val = self.min_max["max"][i]
                value = max(min_val, min(value, max_val))
                # Normalize value
                normalized_value = (value - min_val) / (max_val - min_val)
                filtered_state.append(normalized_value)
            else:
                raise KeyError(f"State is missing required field: {field}")
        return tf.convert_to_tensor(filtered_state, dtype=tf.float32)

    def chooseAction(self, state: dict) -> int:
        """
        Choose an action using epsilon-greedy policy.
        """
        filtered_state = self.filter_and_transform_state(state)
        if isinstance(self.epsilon, EpsilonScheduler):
            epsilon_value = self.epsilon.get_epsilon()
        else:
            epsilon_value = self.epsilon
        if np.random.uniform(0, 1) < epsilon_value:
            # Explore: choose random action
            return np.random.randint(0, self.action_dim)
        else:
            # Exploit: choose the best action
            filtered_state = tf.expand_dims(filtered_state, axis=0)  # Add batch dimension
            q_values = self.model(filtered_state, training=False)
            return tf.argmax(q_values[0]).numpy()

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store an experience tuple in the replay buffer.
        """
        if len(self.replay_buffer) >= self.buffer_capacity:
            self.replay_buffer.pop(0)  # Remove the oldest experience
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_from_replay(self):
        """
        Sample a batch of experiences from the replay buffer and train the network.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = tf.convert_to_tensor([self.filter_and_transform_state(s).numpy() for s in states], dtype=tf.float32)
        next_states = tf.convert_to_tensor([self.filter_and_transform_state(s).numpy() for s in next_states], dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute target Q-values
        next_q_values = self.target_model(next_states, training=False)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        targets = rewards + (1 - dones) * self.discount_factor * max_next_q_values

        # Update the main network
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            action_indices = tf.stack([tf.range(self.batch_size), actions], axis=1)
            predicted_q_values = tf.gather_nd(q_values, action_indices)
            loss = tf.keras.losses.MSE(targets, predicted_q_values)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def update_knowledge(self, state: dict, action: str, reward: float, next_state: dict, done: bool):
        """
        Store the experience in the replay buffer and occasionally train the network.
        """
        self.store_experience(state, Springer.ACTIONS.index(action), reward, next_state, done)
        self.train_from_replay()

    def save_model(self, path):
        """Save the model to the specified path."""
        self.model.save(path)

    def load_model(self, path):
        """Load the model from the specified path."""
        self.model = tf.keras.models.load_model(path)
