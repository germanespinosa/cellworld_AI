from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import cellworld_gym as cwg
import cellworld_belief as belief
import gymnasium


def get_belief_state_components(condition: int):
    NB = belief.NoBeliefComponent()
    LOS = belief.LineOfSightComponent(other_scale=.5)
    V = belief.VisibilityComponent()
    D = belief.DiffusionComponent()
    GD = belief.GaussianDiffusionComponent()
    DD = belief.DirectedDiffusionComponent()
    O = belief.OcclusionsComponent()
    A = belief.ArenaComponent()
    M = belief.MapComponent()
    NL = belief.ProximityComponent()

    components = []
    if condition == 0:
        components = [NL, M]
    if condition == 1:
        components = [NB, LOS, M]
    elif condition == 2:
        components = [V, LOS, M]
    elif condition == 3:
        components = [GD, V, LOS, M]
    elif condition == 4:
        components = [DD, V, LOS, M]
    elif condition == 5:
        components = [DD, GD, V, LOS, M]
    return components


def create_env(world_name: str = "21_05",
               reward_function = None,
               use_lppos: bool = True,
               use_predator: bool = False,
               condition: int = 1,
               max_steps: int = 300,
               time_step: float = .25,
               reward_structure: dict = {},
               render: bool = False,
               real_time: bool = False,
               **kwargs):

    return gymnasium.make("CellworldBotEvadeBelief-v0",
                          world_name=world_name,
                          use_lppos=use_lppos,
                          use_predator=use_predator,
                          max_step=max_steps,
                          time_step=time_step,
                          reward_function=reward_function,
                          real_time=real_time,
                          render=render,
                          belief_state_components=get_belief_state_components(condition=condition))


def create_vec_env(environment_count: int,
                   reward_function,
                   world_name: str = "21_05",
                   use_lppos: bool = False,
                   use_predator: bool = False,
                   condition: int = 1,
                   max_steps: int = 300,
                   time_step: float = .25,
                   **kwargs):

    return DummyVecEnv([lambda:
                        create_env(world_name=world_name,
                                   use_lppos=use_lppos,
                                   use_predator=use_predator,
                                   condition=condition,
                                   max_steps=max_steps,
                                   time_step=time_step,
                                   reward_function=reward_function)
                        for _ in range(environment_count)])
