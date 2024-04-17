from cellworld_game import Reward
from cellworld_gym import BotEvade
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


def create_vec_env(environment_count: int,
                   use_lppos: bool = True,
                   use_predator: bool = False,
                   max_steps: int = 300,
                   step_wait: int = 10,
                   reward_structure: dict = {},
                   **kwargs):
    puse_lppos, puse_predator, pmax_steps, pstep_wait, preward_structure = use_lppos, use_predator, max_steps, step_wait, reward_structure
    return DummyVecEnv([lambda: BotEvade(world_name="21_05",
                                         use_lppos=puse_lppos,
                                         use_predator=puse_predator,
                                         max_step=pmax_steps,
                                         step_wait=pstep_wait,
                                         reward_function=Reward(preward_structure))
                        for _ in range(environment_count)])


