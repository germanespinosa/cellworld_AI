import metrica
metrica.start_profile()
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.append('../../')
import json
import argparse
from env import create_vec_env
from algorightms import algorithms
from callback import CellworldCallback
import config

config.task_name = "botevade"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cellworld AI BotEvade training tool: trains an RL model on the BotEvade Cellworld OpenAI Gym environment')
    parser.add_argument('model_name', type=str, help='name of the model file in the models folder')
    parser.add_argument('-r', '--run_identifier', type=str, help='string identifying the run')
    parser.add_argument('-b', '--replay_buffer_file', type=str, help='replay buffer file', required=False)
    parser.add_argument('-t', '--tlppo', action='store_true', help='performs tlppo training')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

    run_identifier = config.run_identifier(run_id=args.run_identifier)
    model_configuration_file = config.model_config_file(model_name=args.model_name)

    if not os.path.exists(model_configuration_file):
        print(f"Model configuration file '{model_configuration_file}' not found")
        exit(1)

    replay_buffer_file = ""
    if args.replay_buffer_file:
        replay_buffer_file = config.in_buffer_file(replay_buffer_file=replay_buffer_file)
        if not os.path.exists(replay_buffer_file):
            print(f"Replay buffer file '{replay_buffer_file}' not found")
            exit(1)

    model_config = json.loads(open(model_configuration_file).read())

    run_replay_buffer_file = config.out_buffer_file(model_name=args.model_name,
                                                    run_identifier=run_identifier)
    run_data_file = config.data_file(model_name=args.model_name,
                                     run_identifier=run_identifier)
    logs_folder = config.tensor_board_logs_folder(model_name=args.model_name,
                                                  run_identifier=run_identifier)
    tlppo = True if args.tlppo else False
    vec_envs = create_vec_env(use_lppos=tlppo,
                              **model_config)

    print("envs created: ", len(vec_envs.envs))

    algorithm = algorithms[model_config["algorithm"]]

    if os.path.exists(run_data_file):
        print(f"Data file '{run_data_file}' found, loading...")
        model = algorithm.load(env=vec_envs,
                               path=run_data_file)
        reset_num_time_steps = False
    else:
        print(f"Data file '{run_data_file}' not found")
        model = algorithm.create(environment=vec_envs,
                                 tensorboard_log_folder=logs_folder,
                                 **model_config)
        reset_num_time_steps = True

    if args.replay_buffer_file:
        print(f"loading replay buffer file {replay_buffer_file}")
        model.load_replay_buffer(replay_buffer_file)

    model.learn(total_timesteps=model_config["training_steps"],
                log_interval=model_config["log_interval"],
                tb_log_name=logs_folder,
                callback=CellworldCallback(),
                reset_num_timesteps=reset_num_time_steps)

    print(f"saving replay buffer file {run_replay_buffer_file}")
    model.save_replay_buffer(run_replay_buffer_file)

    print(f"saving data file {run_data_file}")
    model.save(run_data_file)
