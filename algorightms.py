from stable_baselines3 import DQN, PPO
import typing
from callback import CellworldCallback
from sb3_contrib import TRPO
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from gymnasium import Env


def DQN_train(environment: VecEnv,
              name: str,
              training_steps: int,
              network_architecture: typing.List[int],
              learning_rate: float,
              log_interval: int,
              batch_size: int,
              learning_starts: int,
              replay_buffer_file: str = "",
              **kwargs: typing.Any):
    model = DQN("MlpPolicy",
                environment,
                verbose=1,
                batch_size=batch_size,
                learning_rate=learning_rate,
                train_freq=(1, "step"),
                buffer_size=training_steps,
                learning_starts=learning_starts,
                replay_buffer_class=ReplayBuffer,
                tensorboard_log="./tensorboard_logs/",
                policy_kwargs={"net_arch": network_architecture}
                )
    custom_callback = CellworldCallback()
    if replay_buffer_file:
        model.load_replay_buffer(replay_buffer_file)
    model.learn(total_timesteps=training_steps,
                log_interval=log_interval,
                tb_log_name=name,
                callback=custom_callback)
    model.save_replay_buffer("models/%s_buffer.pickle" % name)
    model.save("models/%s" % name)


def DQN_show(environment: Env,
             file_name: str):
    model = environment.get_wrapper_attr('model')
    model.render = True
    loaded_model = DQN.load(file_name)
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


def QRDQN_train(environment: VecEnv,
                name: str,
                training_steps: int,
                network_architecture: typing.List[int],
                learning_rate: float,
                log_interval: int,
                batch_size: int,
                learning_starts: int,
                replay_buffer_file: str = "",
                **kwargs: typing.Any):
    model = QRDQN("MlpPolicy",
                  environment,
                  verbose=1,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  train_freq=(1, "step"),
                  buffer_size=training_steps,
                  learning_starts=learning_starts,
                  replay_buffer_class=ReplayBuffer,
                  tensorboard_log="./tensorboard_logs/",
                  policy_kwargs={"net_arch": network_architecture}
                  )
    custom_callback = CellworldCallback()
    if replay_buffer_file:
        model.load_replay_buffer(replay_buffer_file)
    model.learn(total_timesteps=training_steps,
                log_interval=log_interval,
                tb_log_name=name,
                callback=custom_callback)
    model.save("models/%s" % name)


def QRDQN_show(environment: Env,
               file_name: str):
    loaded_model = QRDQN.load(file_name)
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


def PPO_train(environment: VecEnv,
              name: str,
              training_steps: int,
              network_architecture: typing.List[int],
              learning_rate: float,
              log_interval: int,
              n_steps: int,
              replay_buffer_file: str = "",
              **kwargs: typing.Any):
    model = PPO("MlpPolicy",
                environment,
                learning_rate = learning_rate,
                n_steps = n_steps,
                policy_kwargs={"net_arch": network_architecture},
                tensorboard_log="./tensorboard_logs/",
                verbose=1)
    custom_callback = CellworldCallback()
    model.learn(total_timesteps=training_steps,
                log_interval=log_interval,
                tb_log_name=name,
                callback=custom_callback)
    model.save("models/%s" % name)


def PPO_show(environment: Env,
             file_name: str):
    loaded_model = PPO.load(file_name)
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


def RPPO_train(environment: VecEnv,
              name: str,
              training_steps: int,
              network_architecture: typing.List[int],
              learning_rate: float,
              log_interval: int,
              batch_size: int,
              n_steps: int,
              replay_buffer_file: str = "",
              **kwargs: typing.Any):
    model = RecurrentPPO("MlpLstmPolicy",
                         environment,
                         batch_size = batch_size,
                         learning_rate = learning_rate,
                         policy_kwargs={"net_arch": network_architecture},
                         n_steps = n_steps,
                         verbose=1)
    custom_callback = CellworldCallback()
    model.learn(total_timesteps=training_steps,
                log_interval=log_interval,
                tb_log_name=name,
                callback=custom_callback)
    model.save("models/%s" % name)


def RPPO_show(environment: Env,
              file_name: str):
    loaded_model = RecurrentPPO.load(file_name)
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


def TRPO_train(environment: VecEnv,
               name: str,
               training_steps: int,
               network_architecture: typing.List[int],
               learning_rate: float,
               log_interval: int,
               replay_buffer_file: str = "",
               **kwargs: typing.Any):
    model = TRPO("MlpPolicy",
                 environment,
                 verbose=1,
                 tensorboard_log="./tensorboard_logs/",
                 policy_kwargs={"net_arch": network_architecture},
                 learning_rate=learning_rate)
    custom_callback = CellworldCallback()
    model.learn(total_timesteps=training_steps,
                log_interval=log_interval,
                tb_log_name=name,
                callback=custom_callback)
    model.save("models/%s" % name)


def TRPO_show(environment: Env,
              file_name: str):
    loaded_model = TRPO.load(file_name)
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



algorithms = {"DQN": (DQN_train, DQN_show),
              "PPO": (PPO_train, PPO_show),
              "QRDQN": (QRDQN_train, QRDQN_show),
              "TRPO": (TRPO_train, TRPO_show),
              "RPPO": (RPPO_train, RPPO_show)}
