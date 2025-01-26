settings = {

    # General
    "frames_per_generation": 600,
    "springers_per_generation": 1,

    # Graphics
    "first_animation_frame": 100,
    "x_res": 900,
    "y_res": 600,

    # Physics
    "gravity": 0.04,
    "ground_height_fraction": 0.1,
    "epsilon": 0.001,

    # Springers
    "jump_force": 3,
    "side_force": 10,
    "jump_cooldown": 30,
    "shift_cooldown": 8
}

rewards = {
    "all_penalty_frames" : 4,
    "speed_average_duration" : 7,
    "reward_mulitplier" : 2.917,
    "penalty_multiplier": 0.0058,
}
learing = {
    "learning_rate": 0.0064,
    "discount_factor": 0.162,
    "leg_angle_buckets": 13,
    "epsilon_target": 0.051,
    "epsilon_decay": 0.928,
    "epsilon_initial": 0.917,
}

debug = {
         "reward_list": [],
         "score_list": []
         }