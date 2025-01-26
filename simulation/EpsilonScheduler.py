

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