import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import argparse
from tasks.botevade.env import create_env
from algorightms import algorithms
import config

config.task_name = "botevade"


def parse_arguments():
    parser = argparse.ArgumentParser(description='Cellworld AI visualization tool: executes a trained RL model on a Cellworld OpenAI Gym environment')
    parser.add_argument('model_name', type=str, help='name of the model file in the models folder')
    parser.add_argument('-r', '--run_identifier', type=str, help='string identifying the run')
    parser.add_argument('-e', '--episode_count', type=int, help='number of episodes to show')
    parser.add_argument('-c', '--cycle', type=int, help='model cycle')
    parser.add_argument('-v', '--video', action='store_true', help='output episodes videos')
    parser.add_argument('-t', '--tlppo', action='store_true', help='performs tlppo training')
    parser.add_argument('-s', '--silent', action='store_true', help='renders the environment to the screen')
    parser.add_argument('-rt', '--real_time', action='store_true', help='run_in_real_time')

    args = parser.parse_args()
    if args.video and args.silent:
        parser.error('--video_file cannot be used with --draw')

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
    else:
        print(f"Model configuration file '{model_configuration_file}' found")

    run_data_file = config.data_file()

    if args.cycle:
        run_data_file = run_data_file.replace(".zip", f"_{args.cycle}.zip")

    if not os.path.exists(run_data_file):
        print(f"Data file '{run_data_file}' not found")
        exit(1)
    else:
        print(f"Data file '{run_data_file}' found")

    model_config = json.loads(open(model_configuration_file).read())

    environment = create_env(use_lppos=args.tlppo,
                             render=not args.silent,
                             real_time=args.real_time,
                             **model_config)

    algorithm = algorithms[model_config["algorithm"]]
    model = algorithm.load(run_data_file)

    if args.video:
        videos_folder = config.video_folder()
        print(f"Saving videos to {videos_folder}")
        from cellworld_game import save_video_output
        save_video_output(environment.model,
                          video_folder=videos_folder)

    scores = []

    episode_count = 10
    if args.episode_count:
        episode_count = args.episode_count

    for i in range(episode_count):
        observation, _ = environment.reset()
        score, done, tr = 0, False, False
        while not (done or tr):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, tr, _ = environment.step(action)
            score += reward
        scores.append(score)
    environment.close()

