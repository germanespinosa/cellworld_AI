import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import pickle
import json
import typing
import cellworld as cw
import cellworld_game as cwgame
import numpy as np
from collections import namedtuple
from cellworld_gym import BotEvadeEnv
from stable_baselines3.common.buffers import ReplayBuffer
from env import create_env


def parse_arguments():
    parser = argparse.ArgumentParser(description='Cellworld AI replay tool: generates a ReplayBuffer from an existing set of cellworld experiments')
    parser.add_argument('model_name', type=str, help='name of the model file in the models folder')
    parser.add_argument('-bf', '--replay_buffer_folder', type=str, help='folder containing cellworld experiment files', required=True)
    parser.add_argument('-bs', '--replay_buffer_size', type=int, help='size of the replay buffer (default 10000)', required=False)
    parser.add_argument('-bo', '--replay_buffer_output_file', type=str, help='replay buffer output file', required=True)
    parser.add_argument('-fp', '--filter_by_phases', type=str, help='experimental phases that should be included in the buffer (default ALL)', required=False)
    parser.add_argument('-fs', '--filter_by_subject', type=str, help='experimental subjects that should be included in the buffer (default ALL)', required=False)
    parser.add_argument('-s', '--sort_by', type=str, help='sorts the experiments before creating the replay buffer (default NONE)', required=False)
    args = parser.parse_args()
    return args


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
    state.direction = 90-step.rotation
    return state


def get_agent_states_from_episode(episode: cw.Episode,
                                  time_step: float,
                                  actions: cw.Cell_group) -> typing.Tuple[cwgame.AgentState, typing.Optional[cwgame.AgentState]]:

    trajectories = episode.trajectories.split_by_agent()

    if "prey" not in trajectories:
        return

    prey_trajectory = trajectories["prey"]

    prey_step_index = 0

    prey_step = prey_trajectory[prey_step_index]
    step_time = prey_step.time_stamp
    prey_state = step_to_state(prey_trajectory[prey_step_index])

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
        prev_prey_state = prey_state
        prey_state = step_to_state(prey_trajectory[prey_step_index])

        action = actions.find(prey_trajectory[prey_step_index].location)

        if action:
            started = True

        if has_predator:
            predator_state = step_to_state(predator_trajectory.get_step_by_time_stamp(step_time))

        if started:
            prey_state.direction = cwgame.direction(prev_prey_state.location, prey_state.location)
            if has_predator:
                yield {"prey": prey_state, "predator": predator_state}, action
            else:
                yield {"prey": prey_state}, action


def fill_buffer(experiment_file: str,
                buffer: ReplayBuffer,
                env: BotEvadeEnv,
                buffer_size: int):
    experiment = cw.Experiment.load_from_file(experiment_file)
    loader = env.get_wrapper_attr('loader')
    actions = loader.world.cells.free_cells()
    time_step = env.get_wrapper_attr('time_step')
    reset = env.get_wrapper_attr('replay_reset')
    step = env.get_wrapper_attr('replay_step')
    for episode in experiment.episodes:
        prev_observation, infos = None, {}
        for agents_state, action in get_agent_states_from_episode(episode=episode,
                                                                  time_step=time_step,
                                                                  actions=actions):
            if prev_observation is None:
                prev_observation, infos = reset(agents_state=agents_state)
                continue

            post_observation, reward, done, truncated, infos = step(agents_state=agents_state)
            buffer.add(obs=prev_observation,
                       next_obs=post_observation,
                       action=np.array([action]),
                       reward=np.array(reward),
                       done=np.array(done),
                       infos=[infos])
            prev_observation = post_observation

            if done:
                break

            if buffer.size() % 1000 == 0:
                print(f"{buffer.size()} out of {buffer_size} records so far")

            if buffer.size() == buffer_size:
                break


def replay_episode(episode: cw.Episode, env: BotEvadeEnv):
    env.model.prey.max_forward_speed = 0
    env.model.prey.max_turning_speed = 0
    env.model.predator.max_forward_speed = 0
    env.model.predator.max_turning_speed = 0
    for step in episode.trajectories:
        env.step(step.action)


def parse_experiment_file(experiment_file_path: str) -> typing.Optional[namedtuple]:
    import re
    import datetime
    experiment_file_name = os.path.basename(experiment_file_path)
    experiment_name = experiment_file_name.replace("_experiment.json", "")
    ExperimentData = namedtuple("experiment_data",
                                ["name", "prefix", "subject", "phase", "iteration", "date_time", "occlusions", "file_path"])
    parts = experiment_name.split("_")
    prefix = parts[0]
    phase_iteration = parts[-1]
    match = re.match(r'(\D+)(\d+)', phase_iteration)
    if not match:
        return None
    phase = match.group(1)
    iteration = int(match.group(2))
    occlusions = "%s_%s" % (parts[-3], parts[-2])
    subject = parts[-4]
    date_time = datetime.datetime.strptime("%s_%s" % (parts[1], parts[2]), "%Y%m%d_%H%M")
    data = ExperimentData(file_path=experiment_file_path,
                          name=experiment_name,
                          prefix=prefix,
                          phase=phase,
                          iteration=iteration,
                          occlusions=occlusions,
                          subject=subject,
                          date_time=date_time)
    return data


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)


model_file = "models/%s_config.json" % args.model_name

if not os.path.exists(model_file):
    print("Model File not found")
    exit(1)

model_config = json.loads(open(model_file).read())

if os.path.exists(args.replay_buffer_output_file):
    print("Output file already exists")
    exit(1)


if args.replay_buffer_size:
    buffer_size = args.replay_buffer_size
else:
    buffer_size = 10000

print("Loading replay buffer from %s" % args.replay_buffer_folder)

model_config_predator = {}
model_config_no_predator = {}
model_config_predator.update(model_config)
model_config_predator["use_predator"] = True
model_config_no_predator.update(model_config)
model_config_no_predator["use_predator"] = False

pred_env = create_env(use_lppos=False,
                      **model_config_predator)

no_pred_env = create_env(use_lppos=False,
                         **model_config_no_predator)

replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                             observation_space=pred_env.observation_space,
                             action_space=pred_env.action_space)

experiments = [parse_experiment_file(experiment_file_path=file_path) for file_path in experiment_files(start_path=args.replay_buffer_folder)]

experiments = [data for data in experiments if data]

if args.filter_by_phases:
    phases_filter = args.filter_by_phases.split(",")
    experiments = [data for data in experiments if data.phase in phases_filter]

if args.filter_by_subject:
    subject_filter = args.filter_by_subject.split(",")
    experiments = [data for data in experiments if data.subject in subject_filter]

if args.sort_by:
    sort_by = args.sort_by.split(",")
    experiments = sorted(experiments, key=lambda data: tuple([getattr(data, field) for field in sort_by]))


for data in experiments:
    print(f"Loading experiment {data.name}")
    if "R" in data.phase:
        fill_buffer(data.file_path, replay_buffer, pred_env, buffer_size)
    else:
        fill_buffer(data.file_path, replay_buffer, no_pred_env, buffer_size)
    if replay_buffer.size() == buffer_size:
        break

replay_buffer_output_file = f"buffers/{args.replay_buffer_output_file}"

print(f"saving replay buffer file {replay_buffer_output_file}")

with open(replay_buffer_output_file, 'wb') as f:
    pickle.dump(replay_buffer, f)


