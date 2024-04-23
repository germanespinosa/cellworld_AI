from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import cellworld_gym as cwg
import gymnasium


def create_vec_env(environment_count: int,
                   use_lppos: bool = True,
                   use_predator: bool = False,
                   max_steps: int = 300,
                   time_step: float = .25,
                   reward_structure: dict = {},
                   **kwargs):

    return DummyVecEnv([lambda: gymnasium.make("CellworldBotEvade-v0",
                                               world_name="21_05",
                                               use_lppos=use_lppos,
                                               use_predator=use_predator,
                                               max_step=max_steps,
                                               time_step=time_step,
                                               reward_function=cwg.BotEvadeReward(reward_structure))
                        for _ in range(environment_count)])
