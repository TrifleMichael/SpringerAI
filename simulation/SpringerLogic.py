import numpy as np
import settings
from random import uniform, randrange

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
    def __init__(self, height_buckets, leg_angle_buckets, action_number = 4):
        self.knowledge = {}

    def retrieveFromState(self, state: dict, parameter: str) -> str:
        if parameter not in state:
            raise Exception(f"Error: Incorrectly accessign state, parameter {parameter} does not exist in {state}")
        return state[parameter]
        
    def chooseAction(self, state: dict) -> str:
        # Used for choosing action
        q_leg_angle = self.quantize_leg_angle(self.retrieveFromState(state, "leg_angle")) # TODO: Quantize
        height = self.retrieveFromState(state, "height") # TODO: Quantize
        
        marked_for_removal = self.retrieveFromState(state, "marked_for_removal") # Used to check if springer fell over
        step = self.retrieveFromState(state, "step") # Used to check if generation is about to finish

        # TODO: Reinforcment should happen at every step, not just when the simulation ends, right?
        if marked_for_removal or step == settings.settings["frames_per_generation"] - 1:
            # Update knowledge
            x_distance = self.retrieveFromState(state, "x_distance")
            # TODO: Fix line below to fit with this program
            # self.knowledge[tuple(list(quantized_observation) + [action])] = (1-self.learning_rate) * self.knowledge[tuple(list(observation) + [action])] + self.learning_rate * (reward + self.discount * max_reward)
            # TODO: Apply reward proportional to x_distance
        else:
            pass # Predict the next move

    # def quantize_leg_angle(self, leg_angle: float):
        # return (leg_angle - min_leg_angle_val / (max_leg_angle_val - min_leg_angle_val) * bucket_num)

        


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

