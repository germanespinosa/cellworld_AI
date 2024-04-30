import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import argparse
from tasks.botevade.env import create_env
from algorightms import algorithms
import folders

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cellworld AI visualization tool: executes a trained RL model on a Cellworld OpenAI Gym environment')
    parser.add_argument('model_name', type=str, help='name of the model file in the models folder')
    parser.add_argument('-r', '--run_identifier', type=str, help='string identifying the run')
    parser.add_argument('-e', '--episode_count', type=int, help='number of episodes to show')
    parser.add_argument('-v', '--video_file', type=str, help='output video file')
    parser.add_argument('-l', '--lppo', action='store_true', help='use the TLPPO training (default: NO)')
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

    model_config_file = f"{folders.models}/{args.model_name}_config.json"

    if not os.path.exists(model_config_file):
        print(f"Model configuration file '{model_config_file}' not found")
        exit(1)

    if args.run_identifier:
        data_file = f"{folders.data}/{args.model_name}_{args.run_identifier}.zip"
    else:
        data_file = f"{folders.data}/{args.model_name}.zip"

    if not os.path.exists(data_file):
        print(f"Data file '{data_file}' not found")
        exit(1)

    model_config = json.loads(open(model_config_file).read())

    environment = create_env(use_lppos=False,
                             render=True,
                             real_time=False,
                             **model_config)

    algorithm = algorithms[model_config["algorithm"]]
    model = algorithm.load(data_file)

    gameplay_frames = []
    if args.video_file:
        import video
        video.save_video_output(environment=environment,
                                video_file_path=args.video_file)

    scores = []

    episode_count = 10
    if args.episode_count:
        episode_count = args.episode_count

    for i in range(episode_count):
        obs, _ = environment.reset()
        score, done, tr = 0, False, False
        while not (done or tr):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, tr, _ = environment.step(action)
            score += reward
        scores.append(score)
    environment.close()

