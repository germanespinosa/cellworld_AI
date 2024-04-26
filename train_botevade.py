import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import argparse
from env import create_vec_botevade_env
from algorightms import algorithms


def parse_arguments():
    parser = argparse.ArgumentParser(description='Cellworld AI training tool: trains an RL model on a Cellworld OpenAI Gym environment')
    parser.add_argument('model_name', type=str, help='name of the model file in the models folder')
    parser.add_argument('-b', '--replay_buffer_file', type=str, help='replay buffer file', required=False)
    parser.add_argument('-l', '--tlppo', action='store_true', help='performs tlppo training')
    args = parser.parse_args()
    return args


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

    if args.replay_buffer_file and not os.path.exists(args.replay_buffer_file):
        print("Replay buffer File not found")
        exit(1)

    model_config = json.loads(open(model_file).read())

    train, show = algorithms[model_config["algorithm"]]

    vec_envs = create_vec_botevade_env(use_lppos=False,
                                       **model_config)

    train(environment=vec_envs,
          name="%s_control" % args.model_name,
          replay_buffer_file=args.replay_buffer_file,
          **model_config)

    if args.tlppo:
        vec_envs = create_vec_botevade_env(use_lppos=True,
                                           **model_config)

        train(environment=vec_envs,
              name="%s_tlppo" % args.model_name,
              **model_config)


