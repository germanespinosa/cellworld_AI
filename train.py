import os
import json
import sys
from vec_env import create_vec_env
from cellworld_gym.envs.bot_evade import BotEvade
from stable_baselines3 import DQN
from algorightms import algorithms

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def random(environment: BotEvade):
    environment.model.real_time = True
    environment.render_steps = True
    environment.reset()

    for i in range(100000):
        if i % 5 == 0:
            action = environment.action_space.sample()
        obs, reward, done, tr, _ = environment.step(action)
        if environment.prey.finished or i % 200 == 0:
            environment.reset()


def result_visualization(environment: BotEvade,
                         name: str):
    environment.model.real_time = True
    environment.render_steps = True
    loaded_model = DQN.load("models/%s.zip" % name)
    scores = []
    for i in range(100):
        obs, _ = environment.reset()
        score, done, tr = 0, False, False
        while not (done or tr):
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, done, tr, _ = environment.step(action)
            score += reward
        scores.append(score)
    environment.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "-r":
            env = BotEvade(world_name="21_05",
                           use_lppos=True,
                           use_predator=True,
                           max_step=300,
                           step_wait=10)
            random(env)
            exit(0)
        else:
            if len(sys.argv) > 2:
                model_name = sys.argv[2]
            else:
                model_name = input("Model Name: ")

            model_file = "models/%s_config.json" % model_name

            if os.path.exists(model_file):
                model_config = json.loads(open(model_file).read())

                if sys.argv[1] == "-t":
                    vec_envs = create_vec_env(use_lppos=False,
                                              **model_config)

                    algorithms[model_config["algorithm"]](environment=vec_envs,
                                                          name="%s_control" % model_name,
                                                          **model_config)

                    vec_envs = create_vec_env(use_lppos=True,
                                              **model_config)

                    algorithms[model_config["algorithm"]](environment=vec_envs,
                                                          name="%s_tlppo" % model_name,
                                                          **model_config)
                elif sys.argv[1] == "-v":
                    env = BotEvade(world_name="21_05",
                                   use_lppos=False,
                                   use_predator=True,
                                   max_step=300,
                                   step_wait=10)
                    result_visualization(environment=env,
                                         name="%s_tlppo" % model_name)
                exit(0)
            else:
                print("Model File not found")

    else:
        print("Missing parameters")

    print("Usage: python DQN.py <option> <model_name>")
    print("parameters:")
    print("  -t <model_name> : Executes training using the given model configuration file")
    print("  -v <model_name> : Shows visualization of the trained model")
    print("  -r : shows the environment with random policy")
    print()
    exit(1)
