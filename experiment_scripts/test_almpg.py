import omnisafe


if __name__ == "__main__":
    env_id = "SafetyAntVelocity-v1"
    custom_cfgs = {
        "train_cfgs": {
            "total_steps": 100_000,
            "vector_env_nums": 1,
            "device": "cuda:0",
        },
        "algo_cfgs": {
            "steps_per_epoch": 2_000,
            "inner_epoch": 5,
            "critic_update_iters": 3,
            "actor_update_iters": 3,
            "batch_size": 2_000,
            "eta": 1,
        },
        "logger_cfgs": {
            "use_wandb": False,
        },
    }
    agent = omnisafe.Agent("ALMPG", env_id, custom_cfgs=custom_cfgs)
    agent.learn()
    agent.evaluate(num_episodes=5)
