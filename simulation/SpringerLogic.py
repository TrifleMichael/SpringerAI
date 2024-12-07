class SpringerLogic:
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
