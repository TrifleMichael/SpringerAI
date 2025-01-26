import optuna
import pygame
import sys
import math
import numpy as np
from EpsilonScheduler import EpsilonScheduler
from simulationManager import SimulationManager
from SpringerLogic import SpringerLogic_QLearning_v2
import settings
from plotting import plot_rewards_and_scores

def run_simulation(trial):
    # Suggest hyperparameters using Optuna
    rewards = {
        "all_penalty_frames": trial.suggest_int("all_penalty_frames", 1, 10),
        "speed_average_duration": trial.suggest_int("speed_average_duration", 1, 10),
        "reward_mulitplier": trial.suggest_float("reward_mulitplier", 1.0, 10.0),
        "penalty_multiplier": trial.suggest_float("penalty_multiplier", 0.0001, 0.01),
    }

    learing = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "discount_factor": trial.suggest_float("discount_factor", 0.01, 0.99),
        "leg_angle_buckets": trial.suggest_int("leg_angle_buckets", 3, 15),
        "epsilon_target": trial.suggest_float("epsilon_target", 0.01, 0.1),
        "epsilon_decay": trial.suggest_float("epsilon_decay", 0.9, 0.9999),
        "epsilon_initial": trial.suggest_float("epsilon_initial", 0.1, 1.0),
    }

    settings.rewards = rewards
    settings.learing = learing

    def single_run():
        pygame.init()
        WIDTH, HEIGHT = settings.settings["x_res"], settings.settings["y_res"]
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("SpringersAI")

        EPSILON = EpsilonScheduler(
            learing["epsilon_initial"], learing["epsilon_target"], learing["epsilon_decay"]
        )

        springer_logic = SpringerLogic_QLearning_v2(
            leg_angle_range=(0, math.pi),
            leg_angle_buckets=learing["leg_angle_buckets"],
            learning_rate=learing["learning_rate"],
            discount_factor=learing["discount_factor"],
            epsilon=EPSILON,
        )

        sim_manager = SimulationManager(screen, springer_logic)
        sim_manager.spawn_springers(1)

        generations = 700  # Number of generations for evaluation per trial
        run_animation = False  # Disable animation for faster optimization

        score_list = []

        for generation in range(generations):
            sim_manager.simulate_generation(run_animation)
            EPSILON.decay()

            # Collect scores
            score_list.extend(settings.debug["score_list"])

        # Quit pygame after the simulation
        pygame.quit()

        # Use the average of the last X scores as the result for this run
        x = 100  # Number of scores to average
        if len(score_list) < x:
            return float("-inf")  # Penalize trials that don't generate enough scores
        
        return np.mean(score_list[-x:])

    # Perform multiple retries and average the results
    retries = 2
    results = [single_run() for _ in range(retries)]
    return np.mean(results)  # Maximize the average result of retries


if __name__ == "__main__":
    # Database URL for PostgreSQL
    storage_url = "postgresql://optuna_user:optuna_pass@localhost:5432/optuna_db"

    # Create Optuna study
    study = optuna.create_study(direction="maximize", storage=storage_url, study_name="springers_study", load_if_exists=True)
    study.optimize(run_simulation, n_trials=60)  # Number of trials to run

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters for further use
    best_params = study.best_trial.params

    # Dump best parameters to a .py file
    with open("best_hyperparameters.py", "w") as f:
        f.write("# Best hyperparameters from Optuna\n")
        f.write(f"best_params = {best_params}\n")

    print("Best hyperparameters saved to best_hyperparameters.py")

