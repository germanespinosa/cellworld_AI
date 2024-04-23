import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import argparse
from stable_baselines3.common.buffers import ReplayBuffer
from vec_env import create_vec_env
from algorightms import algorithms
from replay import experiment_files, fill_buffer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Cellworld AI training tool: trains an RL model on a Cellworld OpenAI Gym environment')
    parser.add_argument('model_name', type=str, help='name of the model file in the models folder')
    parser.add_argument('-b', '--replay_buffer_folder', type=str, help='folder containing cellworld experiment files', required=False)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_arguments()
        model_file = "models/%s_config.json" % args.model_name

        if os.path.exists(model_file):
            model_config = json.loads(open(model_file).read())

            if args.replay_buffer_folder:
                print("Loading replay buffer from %s" % args.replay_buffer_folder)
                env = create_vec_env(use_lppos=False,
                                     **model_config).envs[0]
                buffer = ReplayBuffer(buffer_size=1_000_000,
                                      observation_space=env.observation_space,
                                      action_space=env.action_space)
                for experiment_file in experiment_files(start_path=args.replay_buffer_folder):
                    file_name = os.path.basename(experiment_file)
                    print(f"Loading experiment file {file_name}")
                    fill_buffer(experiment_file, buffer, env)

            vec_envs = create_vec_env(use_lppos=False,
                                      **model_config)

            train, show = algorithms[model_config["algorithm"]]

            train(environment=vec_envs,
                  name="%s_control" % args.model_name,
                  **model_config)

            vec_envs = create_vec_env(use_lppos=True,
                                      **model_config)

            train(environment=vec_envs,
                  name="%s_tlppo" % args.model_name,
                  **model_config)
        else:
            print("Model File not found")
            exit(1)

    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

