import os
import typing
import cellworld as cw
import cellworld_game as cwgame
import numpy as np
from cellworld_gym import BotEvadeEnv
from stable_baselines3.common.buffers import ReplayBuffer


def experiment_files(start_path: str) -> typing.List[str]:
    entries = os.listdir(start_path)
    for entry in entries:
        full_path = os.path.join(start_path, entry)
        if os.path.isdir(full_path):
            experiment_file = f"{entry}_experiment.json"
            experiment_file_path = os.path.join(full_path, experiment_file)
            if os.path.exists(experiment_file_path):
                yield experiment_file_path


def step_to_state(step: cw.Step) -> cwgame.AgentState:
    state = cwgame.AgentState()
    state.location = tuple(step.location.get_values())
    state.direction = step.rotation
    return state


def get_agent_states_from_episode(episode: cw.Episode,
                                  time_step: float,
                                  actions: cw.Cell_group) -> typing.Tuple[cwgame.AgentState, typing.Optional[cwgame]]:

    trajectories = episode.trajectories.split_by_agent()

    if "prey" not in trajectories:
        return

    prey_trajectory = trajectories["prey"]

    prey_step_index = 0

    prey_step = prey_trajectory[prey_step_index]
    step_time = prey_step.time_stamp

    has_predator = "predator" in trajectories
    if has_predator:
        predator_trajectory = trajectories["predator"]
        predator_state = predator_trajectory.get_step_by_time_stamp(step_time)
    else:
        predator_trajectory = cw.Trajectories()
        predator_state = None

    started = False

    while prey_step_index < len(prey_trajectory):
        step_time += time_step

        while prey_step_index < len(prey_trajectory) and prey_trajectory[prey_step_index].time_stamp < step_time:
            prey_step_index += 1

        if prey_step_index == len(prey_trajectory):
            break

        prey_state = step_to_state(prey_trajectory[prey_step_index])
        action = actions.find(prey_trajectory[prey_step_index].location)

        if action:
            started = True

        if has_predator:
            predator_state = step_to_state(predator_trajectory.get_step_by_time_stamp(step_time))
        if started:
            yield prey_state, predator_state, action


def fill_buffer(experiment_file: str,
                buffer: ReplayBuffer,
                env: BotEvadeEnv,
                buffer_size: int):
    experiment = cw.Experiment.load_from_file(experiment_file)
    loader = env.get_wrapper_attr('loader')
    actions = loader.world.cells.free_cells()
    time_step = env.get_wrapper_attr('time_step')
    get_observation = env.get_wrapper_attr('get_observation')
    model = env.get_wrapper_attr('model')
    for episode in experiment.episodes:
        prev_observation, reward, done, truncated, infos = None, 0, False, False, {}
        for prey_state, predator_state, action in get_agent_states_from_episode(episode=episode,
                                                                                time_step=time_step,
                                                                                actions=actions):
            model.has_predator = predator_state is not None

            if prev_observation is None:
                post_observation, infos = env.reset()
            else:
                post_observation, reward, done, truncated, infos = env.step(action)
            post_observation = get_observation()
            model.prey.set_state(prey_state)
            if predator_state is not None:
                model.predator.set_state(predator_state)
            if prev_observation is not None:
                buffer.add(obs=prev_observation,
                           next_obs=post_observation,
                           action=np.array([action]),
                           reward=np.array(reward),
                           done=np.array(done),
                           infos=[infos])
            if done:
                break

            if buffer.size() % 100 == 0:
                print(f"{buffer.size()} out of {buffer_size} records so far")

            if buffer.size() == buffer_size:
                break
            prev_observation = post_observation






