import metrica
metrica.start_profile()
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.append('../../')
import json
import argparse
from env import create_vec_env, set_other_policy
from algorightms import algorithms
from callback import CellworldCallback
import config
import cellworld_gym as cwg

config.task_name = "dualevade"

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

    run_identifier_1 = "mouse_1_" + config.run_identifier(run_id=args.run_identifier)
    run_identifier_2 = "mouse_2_" + config.run_identifier(run_id=args.run_identifier)

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

    run_replay_buffer_file_1 = config.out_buffer_file(model_name=args.model_name,
                                                      run_identifier=run_identifier_1)
    run_replay_buffer_file_2 = config.out_buffer_file(model_name=args.model_name,
                                                      run_identifier=run_identifier_2)

    run_data_file_1 = config.data_file(model_name=args.model_name,
                                       run_identifier=run_identifier_1)
    run_data_file_2 = config.data_file(model_name=args.model_name,
                                       run_identifier=run_identifier_2)

    logs_folder_1 = config.tensor_board_logs_folder(model_name=args.model_name,
                                                    run_identifier=run_identifier_1)
    logs_folder_2 = config.tensor_board_logs_folder(model_name=args.model_name,
                                                    run_identifier=run_identifier_2)

    tlppo = True if args.tlppo else False
    vec_envs_1 = create_vec_env(use_lppos=tlppo,
                                pov=cwg.DualEvadePov.mouse_1,
                                **model_config)

    print("Mouse 1 envs created: ", len(vec_envs_1.envs))

    vec_envs_2 = create_vec_env(use_lppos=tlppo,
                                pov=cwg.DualEvadePov.mouse_2,
                                **model_config)

    print("Mouse 2 envs created: ", len(vec_envs_2.envs))

    algorithm = algorithms[model_config["algorithm"]]

    if os.path.exists(run_data_file_1):
        print(f"Data file '{run_data_file_1}' found, loading...")
        model_1 = algorithm.load(env=vec_envs_1,
                                 path=run_data_file_1)

        reset_num_time_steps_1 = False
    else:
        print(f"Data file '{run_data_file_1}' not found")
        model_1 = algorithm.create(environment=vec_envs_1,
                                   tensorboard_log_folder=logs_folder_1,
                                   **model_config)

        reset_num_time_steps_1 = True


    if os.path.exists(run_data_file_2):
        print(f"Data file '{run_data_file_2}' found, loading...")
        model_2 = algorithm.load(env=vec_envs_2,
                                 path=run_data_file_2)

        reset_num_time_steps_2 = False
    else:
        print(f"Data file '{run_data_file_2}' not found")
        model_2 = algorithm.create(environment=vec_envs_2,
                                   tensorboard_log_folder=logs_folder_2,
                                   **model_config)

        reset_num_time_steps_2 = True


    set_other_policy(vec_env=vec_envs_1, model=model_2)
    set_other_policy(vec_env=vec_envs_2, model=model_1)

    for episode in range(model_config["training_cycles"]):
        model_1.learn(total_timesteps=model_config["training_steps"],
                      log_interval=model_config["log_interval"],
                      tb_log_name=logs_folder_1,
                      callback=CellworldCallback(),
                      reset_num_timesteps=reset_num_time_steps_1)

        reset_num_time_steps_1 = False

        model_2.learn(total_timesteps=model_config["training_steps"],
                      log_interval=model_config["log_interval"],
                      tb_log_name=logs_folder_2,
                      callback=CellworldCallback(),
                      reset_num_timesteps=reset_num_time_steps_2)

        reset_num_time_steps_2 = False

        print(f"saving data file {run_data_file_1}")
        model_1.save(run_data_file_1)
        print(f"saving data file {run_data_file_2}")
        model_2.save(run_data_file_2)

    print(f"saving replay buffer file {run_replay_buffer_file_1}")
    model_1.save_replay_buffer(run_replay_buffer_file_1)
    print(f"saving replay buffer file {run_replay_buffer_file_2}")
    model_2.save_replay_buffer(run_replay_buffer_file_2)

