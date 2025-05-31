import omnisafe


if __name__ == "__main__":
    env_id = "SafetyPointButton1-v0"
    custom_cfgs = {
        "train_cfgs": {
            "total_steps": 1_000_000,
            "vector_env_nums": 1,
            "device": "cuda",
        },
        "algo_cfgs": {
            "steps_per_epoch": 2_000,
            "inner_epoch": 5,
            "critic_update_iters": 3,
            "actor_update_iters": 3,
            "batch_size": 2_000,
            "eta": 0.1,
            "use_armijo_actor": True,
            "use_armijo_critic": True,
            "armijo_alpha_max": 100,
        },
        "logger_cfgs": {
            "use_wandb": False,
        },
    }
    agent = omnisafe.Agent("SPMA", env_id)
    agent.learn()
