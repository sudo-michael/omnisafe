import omnisafe


if __name__ == '__main__':
    env_id = 'SafetyPointGoal1-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 100_000,
            'vector_env_nums': 4,
            'parallel': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': 20_000,
            'update_iters': 1,
        },
        'logger_cfgs': {
            'use_wandb': False,
        },
    }

    agent = omnisafe.Agent('APPO', env_id, custom_cfgs=custom_cfgs)
    agent.learn()
    agent.evaluate(num_episodes=5)
