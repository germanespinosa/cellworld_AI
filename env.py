import typing
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import cellworld_gym as cwg
import gymnasium


def create_botevade_env(world_name: str = "21_05",
                        use_lppos: bool = True,
                        use_predator: bool = False,
                        max_steps: int = 300,
                        time_step: float = .25,
                        reward_structure: dict = {},
                        render: bool = False,
                        real_time: bool = False,
                        **kwargs):

    return gymnasium.make("CellworldBotEvade-v0",
                          world_name=world_name,
                          use_lppos=use_lppos,
                          use_predator=use_predator,
                          max_step=max_steps,
                          time_step=time_step,
                          reward_function=cwg.Reward(reward_structure),
                          real_time=real_time,
                          render=render)


def create_vec_botevade_env(environment_count: int,
                            world_name: str = "21_05",
                            use_lppos: bool = True,
                            use_predator: bool = False,
                            max_steps: int = 300,
                            time_step: float = .25,
                            reward_structure: dict = {},
                            **kwargs):

    return DummyVecEnv([lambda:
                        create_botevade_env(world_name=world_name,
                                            use_lppos=use_lppos,
                                            use_predator=use_predator,
                                            max_steps=max_steps,
                                            time_step=time_step,
                                            reward_structure=reward_structure)
                        for _ in range(environment_count)])


def create_oasis_env(world_name: str = "oasis_island7_02",
                     goal_locations: typing.List[typing.Tuple[float, float]] = None,
                     use_lppos: bool = True,
                     use_predator: bool = False,
                     max_steps: int = 300,
                     time_step: float = .25,
                     reward_structure: dict = {},
                     **kwargs):

    return gymnasium.make("CellworldOasis-v0",
                          world_name=world_name,
                          goal_locations=goal_locations,
                          use_lppos=use_lppos,
                          use_predator=use_predator,
                          max_step=max_steps,
                          time_step=time_step,
                          reward_function=cwg.Reward(reward_structure))


def create_vec_oasis_env(environment_count: int,
                         world_name: str = "oasis_island7_02",
                         goal_locations: typing.List[typing.Tuple[float, float]] = None,
                         use_lppos: bool = True,
                         use_predator: bool = False,
                         max_steps: int = 300,
                         time_step: float = .25,
                         reward_structure: dict = {},
                         **kwargs):

    return DummyVecEnv([lambda:
                        create_oasis_env(world_name=world_name,
                                         goal_locations=goal_locations,
                                         use_lppos=use_lppos,
                                         use_predator=use_predator,
                                         max_steps=max_steps,
                                         time_step=time_step,
                                         reward_structure=reward_structure)
                        for _ in range(environment_count)])

