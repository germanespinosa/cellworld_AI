import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import argparse
from env import create_botevade_env
from algorightms import algorithms


def parse_arguments():
    parser = argparse.ArgumentParser(description='Cellworld AI visualization tool: executes a trained RL model on a Cellworld OpenAI Gym environment')
    parser.add_argument('model_name', type=str, help='name of the model file in the models folder')
    parser.add_argument('-l', '--lppo', action='store_true', help='use the TLPPO training (default: NO)')
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_arguments()
        model_config_file = "models/%s_config.json" % args.model_name
        if args.lppo:
            model_file = "models/%s_tlppo.zip" % args.model_name
        else:
            model_file = "models/%s_control.zip" % args.model_name

        if os.path.exists(model_file):
            model_config = json.loads(open(model_config_file).read())
            env = create_botevade_env(use_lppos=False,
                                      render=True,
                                      real_time=True,
                                      **model_config)

            train, show = algorithms[model_config["algorithm"]]

            show(environment=env,
                 file_name=model_file)

        else:
            print("Model File not found")
            exit(1)

    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)
