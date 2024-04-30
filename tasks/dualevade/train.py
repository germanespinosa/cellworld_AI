import os

import cellworld_gym

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.append('../../')
import json
import argparse
from env import create_vec_env, set_other_policy
from algorightms import algorithms
from callback import CellworldCallback
import folders

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cellworld AI DualEvade training tool: trains an RL model on the DualEvade Cellworld OpenAI Gym environment')
    parser.add_argument('model_name', type=str, help='name of the model file in the models folder')
    parser.add_argument('-r', '--run_identifier', type=str, help='string identifying the run')
    parser.add_argument('-b', '--replay_buffer_file', type=str, help='replay buffer file', required=False)
    parser.add_argument('-tl', '--tlppo', action='store_true', help='performs tlppo training')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

    model_configuration_file = f"{folders.models}/{args.model_name}_config.json"

    if not os.path.exists(model_configuration_file):
        print(f"Model configuration file '{model_configuration_file}' not found")
        exit(1)

    replay_buffer_file = ""
    if args.replay_buffer_file:
        replay_buffer_file = f"{folders.buffers}/{args.replay_buffer_file}"
        if not os.path.exists(replay_buffer_file):
            print(f"Replay buffer file '{replay_buffer_file}' not found")
            exit(1)

    model_config = json.loads(open(model_configuration_file).read())

    run_replay_buffer_file_1 = f"{folders.buffers}/{args.model_name}_{args.run_identifier}_buffer_mouse_1.pickle"
    run_data_file_1 = f"{folders.data}/{args.model_name}_{args.run_identifier}_mouse_1"

    run_replay_buffer_file_2 = f"{folders.buffers}/{args.model_name}_{args.run_identifier}_buffer_mouse_2.pickle"
    run_data_file_2 = f"{folders.data}/{args.model_name}_{args.run_identifier}_mouse_2"

    vec_envs_1 = create_vec_env(use_lppos=args.tlppo,
                                pov=cellworld_gym.DualEvadePov.mouse_1,
                                **model_config)

    vec_envs_2 = create_vec_env(use_lppos=args.tlppo,
                                pov=cellworld_gym.DualEvadePov.mouse_2,
                                **model_config)

    if args.run_identifier:
        logs_folder_1 = f"{args.model_name}_{args.run_identifier}_mouse1"
        logs_folder_2 = f"{args.model_name}_{args.run_identifier}_mouse2"
    else:
        from datetime import datetime
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%Y%m%d_%H%M%S")
        logs_folder_1 = f"{args.model_name}_{formatted_date}_mouse1"
        logs_folder_2 = f"{args.model_name}_{formatted_date}_mouse2"

    algorithm = algorithms[model_config["algorithm"]]

    if os.path.exists(run_data_file_1):
        model_1 = algorithm.load(env=vec_envs_1,
                                 path=run_data_file_1)
        reset_num_time_steps_1 = False
    else:
        model_1 = algorithm.create(environment=vec_envs_1,
                                   tensorboard_log_folder=logs_folder_1,
                                   **model_config)
        reset_num_time_steps_1 = True

    if os.path.exists(run_data_file_2):
        model_2 = algorithm.load(env=vec_envs_2,
                                 path=run_data_file_2)
        reset_num_time_steps_2 = False
    else:
        model_2 = algorithm.create(environment=vec_envs_2,
                                   tensorboard_log_folder=logs_folder_2,
                                   **model_config)
        reset_num_time_steps_2 = True


    set_other_policy(vec_env=vec_envs_1, model=model_2)
    set_other_policy(vec_env=vec_envs_2, model=model_1)

    if args.replay_buffer_file:
        print(f"loading replay buffer file {replay_buffer_file}")
        model_1.load_replay_buffer(replay_buffer_file)
        model_2.load_replay_buffer(replay_buffer_file)

    turns = 10
    if "training_cycles" in model_config:
        turns = model_config["training_cycles"]

    callback_1 = CellworldCallback()
    callback_2 = CellworldCallback()
    for i in range(turns):
        print(f"Starting turn {i+1} of {turns}")

        model_1.learn(total_timesteps=model_config["training_steps"],
                      reset_num_timesteps=reset_num_time_steps_1,
                      log_interval=model_config["log_interval"],
                      callback=callback_1)

        model_2.learn(total_timesteps=model_config["training_steps"],
                      reset_num_timesteps=reset_num_time_steps_2,
                      log_interval=model_config["log_interval"],
                      callback=callback_2)

        reset_num_time_steps_1 = False
        reset_num_time_steps_2 = False

    print(f"saving replay buffer file {run_replay_buffer_file_1}")
    model_1.save_replay_buffer(run_replay_buffer_file_1)

    print(f"saving data file {run_data_file_1}")
    model_1.save(run_data_file_1)

    print(f"saving replay buffer file {run_replay_buffer_file_2}")
    model_1.save_replay_buffer(run_replay_buffer_file_2)

    print(f"saving data file {run_data_file_2}")
    model_2.save(run_data_file_2)

