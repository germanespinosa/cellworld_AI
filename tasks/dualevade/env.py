import gym
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import cellworld_gym as cwg
import gymnasium


def create_env(world_name: str = "21_05",
               use_lppos: bool = True,
               use_other: bool = True,
               use_predator: bool = False,
               max_steps: int = 300,
               time_step: float = .25,
               reward_structure: dict = {},
               render: bool = False,
               real_time: bool = False,
               end_on_pov_goal: bool = True,
               **kwargs):

    env_ = gymnasium.make("CellworldDualEvade-v0",
                          world_name=world_name,
                          use_lppos=use_lppos,
                          use_other=use_other,
                          use_predator=use_predator,
                          max_step=max_steps,
                          time_step=time_step,
                          reward_function=cwg.Reward(reward_structure),
                          real_time=real_time,
                          render=render,
                          end_on_pov_goal=end_on_pov_goal
                          )
    if use_other:
        print(f"Including other agent info in the observation: {type(env_.observation)}")
    else:
        print(f"NOT including other agent info in the observation: {type(env_.observation)}")
    return env_



def create_vec_env(environment_count: int,
                   world_name: str = "21_05",
                   use_lppos: bool = True,
                   use_other: bool = True,
                   use_predator: bool = False,
                   max_steps: int = 300,
                   time_step: float = .25,
                   reward_structure: dict = {},
                   **kwargs):

    return DummyVecEnv([lambda:
                        create_env(world_name=world_name,
                                   use_lppos=use_lppos,
                                   use_other=use_other,
                                   use_predator=use_predator,
                                   max_steps=max_steps,
                                   time_step=time_step,
                                   reward_structure=reward_structure)
                        for _ in range(environment_count)])


def load_vec_env(environment_count: int,
                 world_name: str = "21_05",
                 use_lppos: bool = True,
                 use_other: bool = True,
                 use_predator: bool = False,
                 max_steps: int = 300,
                 time_step: float = .25,
                 reward_structure: dict = {},
                 **kwargs):

    return DummyVecEnv([lambda:
                        create_env(world_name=world_name,
                                   use_lppos=use_lppos,
                                   use_other=use_other,
                                   use_predator=use_predator,
                                   max_steps=max_steps,
                                   time_step=time_step,
                                   reward_structure=reward_structure)
                        for _ in range(environment_count)])


def set_other_policy(vec_env: DummyVecEnv, model):

    def other_policy(obs: cwg.DualEvadeObservation) -> int:
        action, _states = model.predict(obs, deterministic=True)
        return action

    if hasattr(vec_env, "envs"):
        for env in vec_env.envs:
            env.set_other_policy(other_policy)
    else:
        vec_env.set_other_policy(other_policy)

