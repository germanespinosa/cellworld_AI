import numpy as np
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import cellworld_gym as cwg
import cellworld_tlppo as ct
import gymnasium


def on_episode_end(env: ct.TlppoWrapper):
    import matplotlib.pyplot as plt
    lppo, adj_matrix = env.get_lppo()
    plt.figure(figsize=(8, 8))
    plt.scatter(lppo[:, 0], lppo[:, 1], color='blue', zorder=2)  # Plot nodes


    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1:
                plt.plot([lppo[i][0], lppo[j][0]], [lppo[i][1], lppo[j][1]], color='black', zorder=1)

    plt.title('Graph Visualization')
    plt.grid(True)
    plt.show()


def create_env(world_name: str = "21_05",
               use_lppos: bool = True,
               use_predator: bool = False,
               max_steps: int = 300,
               time_step: float = .25,
               reward_structure: dict = {},
               render: bool = False,
               real_time: bool = False,
               **kwargs):

    return gymnasium.make("TlppoWrapper-v0",
                          environment_name="CellworldBotEvade-v0",
                          on_episode_end=on_episode_end,
                          tlppo_dim=np.array([True, True, False, False, False, False, False, False, False, False, False]),
                          tlppo_count=100,
                          world_name=world_name,
                          use_lppos=use_lppos,
                          use_predator=use_predator,
                          max_step=max_steps,
                          time_step=time_step,
                          reward_function=cwg.Reward(reward_structure),
                          real_time=real_time,
                          render=render)


def create_vec_env(environment_count: int,
                   world_name: str = "21_05",
                   use_lppos: bool = False,
                   use_predator: bool = False,
                   max_steps: int = 300,
                   time_step: float = .25,
                   reward_structure: dict = {},
                   **kwargs):

    return DummyVecEnv([lambda:
                        create_env(world_name=world_name,
                                   use_lppos=use_lppos,
                                   use_predator=use_predator,
                                   max_steps=max_steps,
                                   time_step=time_step,
                                   reward_structure=reward_structure)
                        for _ in range(environment_count)])
