import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import typing
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
    parser = argparse.ArgumentParser(description='Cellworld AI BotEvade training tool: trains an RL model on the Cellworld BotEvade OpenAI Gym environment')
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

    config.set_task("botevade")
    config.set_model(args.model_name)
    config.set_run_identifier(args.run_identifier)

    model_configuration_file = config.model_config_file()

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

    run_replay_buffer_file = config.out_buffer_file()
    run_data_file = config.data_file()
    logs_folder = config.tensor_board_logs_folder()
    tlppo = True if args.tlppo else False
    vec_envs = create_vec_env(use_lppos=tlppo,
                              **model_config)

    print("envs created: ", len(vec_envs.envs))

    algorithm = algorithms[model_config["algorithm"]]

    if os.path.exists(run_data_file):
        print(f"Data file '{run_data_file}' found, loading...")
        model = algorithm.load(env=vec_envs,
                               tensorboard_log=logs_folder,
                               path=run_data_file)
        reset_num_time_steps = False
    else:
        print(f"Data file '{run_data_file}' not found")
        model = algorithm.create(environment=vec_envs,
                                 tensorboard_log=logs_folder,
                                 **model_config)
        reset_num_time_steps = True

    if args.replay_buffer_file:
        print(f"loading replay buffer file {replay_buffer_file}")
        model.load_replay_buffer(replay_buffer_file)

    if "training_cycles" in model_config:
        training_cycles = model_config["training_cycles"]
    else:
        training_cycles = 1

    callback = CellworldCallback()

    performance: typing.List[float] = []
    performance_file = config.performance_file()

    if os.path.exists(performance_file):
        with open(performance_file) as f:
            performance = json.load(f)

    best_survival_rate = max(callback.survival)
    cycle_offset = len(performance)

    for cycle in range(training_cycles):
        model.learn(total_timesteps=model_config["training_steps"],
                    callback=callback,
                    reset_num_timesteps=reset_num_time_steps)
        reset_num_time_steps = False

        performance.append(callback.survival)
        if callback.survival > best_survival_rate:
            best_survival_rate = callback.survival
            model.save(run_data_file.replace(".zip", f"_best.zip"))

        print(f"saving data file {run_data_file}")
        model.save(run_data_file)
        model.save(run_data_file.replace(".zip", f"_{cycle + cycle_offset}.zip"))

    if hasattr(model, "save_replay_buffer"):
        print(f"saving replay buffer file {run_replay_buffer_file}")
        model.save_replay_buffer(run_replay_buffer_file)
