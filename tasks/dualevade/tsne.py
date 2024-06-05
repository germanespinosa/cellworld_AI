from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.append('../../')
import json
import argparse
from env import create_vec_env, set_other_policy
from algorightms import algorithms
import config


def parse_arguments():
    parser = argparse.ArgumentParser(description='Cellworld AI BotEvade training tool: trains an RL model on the Cellworld DualEvade OpenAI Gym environment')
    parser.add_argument('model_name', type=str, help='name of the model file in the models folder')
    parser.add_argument('-r', '--run_identifier', type=str, help='string identifying the run')
    parser.add_argument('-b', '--replay_buffer_file', type=str, help='replay buffer file', required=False)
    parser.add_argument('-t', '--tlppo', action='store_true', help='performs tlppo training')
    parser.add_argument('-o', '--other', action='store_true', help='include information about the other agent in observation')
    args = parser.parse_args()
    return args


def get_features(algo, observation):
    import torch
    with torch.no_grad():
        features = algo.policy.q_net.features_extractor(torch.tensor(observation, dtype=torch.float32))
        return features.cpu().numpy()


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except argparse.ArgumentError as e:
        print("Error:", e)
        exit(1)

    config.set_task("dualevade")
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
    model_config["environment_count"] = 1

    run_replay_buffer_file_1 = config.out_buffer_file(suffix="mouse_1")
    with open(run_replay_buffer_file_1, 'rb') as f:
        replay_buffer_1 = pickle.load(f)

    sampled_obs_1 = replay_buffer_1.sample(100000)[0]

    run_replay_buffer_file_2 = config.out_buffer_file(suffix="mouse_2")
    with open(run_replay_buffer_file_2, 'rb') as f:
        replay_buffer_2 = pickle.load(f)

    sampled_obs_2 = replay_buffer_2.sample(100000)[0]

    run_data_file_1 = config.data_file(suffix="mouse_1")
    if not os.path.exists(run_data_file_1):
        print(f"Data file '{run_data_file_1}' not found")
        exit(1)

    run_data_file_2 = config.data_file(suffix="mouse_2")
    if not os.path.exists(run_data_file_2):
        print(f"Data file '{run_data_file_2}' not found")
        exit(1)

    tlppo = True if args.tlppo else False
    other = True if args.other else False

    vec_envs_1 = create_vec_env(use_lppos=tlppo,
                                use_other=other,
                                **model_config)

    print("Mouse 1 envs created: ", len(vec_envs_1.envs))

    vec_envs_2 = create_vec_env(use_lppos=tlppo,
                                use_other=other,
                                **model_config)

    print("Mouse 2 envs created: ", len(vec_envs_2.envs))

    algorithm = algorithms[model_config["algorithm"]]


    print(f"Data file '{run_data_file_1}' found, loading...")
    model_1 = algorithm.load(env=vec_envs_1,
                             path=run_data_file_1)

    print(f"Data file '{run_data_file_2}' found, loading...")
    model_2 = algorithm.load(env=vec_envs_2,
                             path=run_data_file_2)

    set_other_policy(vec_env=vec_envs_1, model=model_2)
    set_other_policy(vec_env=vec_envs_2, model=model_1)

    features1 = get_features(model_1, sampled_obs_1)
    features2 = get_features(model_2, sampled_obs_2)

    other_info_dimensions_1 = sampled_obs_1[:, 3:5].cpu()
    other_info_dimensions_2 = sampled_obs_2[:, 3:5].cpu()

    combined_data_1 = np.concatenate((other_info_dimensions_1, features1), axis=1)
    combined_data_2 = np.concatenate((other_info_dimensions_2, features2), axis=1)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)

    reduced_features_1 = tsne.fit_transform(combined_data_1)
    reduced_features_2 = tsne.fit_transform(combined_data_2)

    plt.figure(figsize=(30, 24))
    plt.scatter(reduced_features_1[:, 0], reduced_features_1[:, 1], c='blue', marker='o')

    # Add titles and labels
    plt.title('t-SNE Visualization with Other Information from Replay Buffer')
    plt.xlabel('t-SNE Other X')
    plt.ylabel('t-SNE Other Y')

    # Show the plot
    plt.grid(True)
    plt.show()

    d = other_info_dimensions_1-other_info_dimensions_2
    colors = (d[:, 0] ** 2 + d[:, 1] ** 2) ** .5

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_features_2[:, 0],
                reduced_features_2[:, 1],
                c=colors,
                marker='.',
                cmap='seismic')

    # Add titles and labels
    plt.title('t-SNE Visualization with Other Information from Replay Buffer')
    plt.xlabel('t-SNE Other X')
    plt.ylabel('t-SNE Other Y')

    # Show the plot
    plt.grid(True)
    plt.show()

