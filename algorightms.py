from stable_baselines3 import DQN, HER
import typing
from callback import CellworldCallback
from sb3_contrib.qrdqn import QRDQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def DQN_train(environment: VecEnv,
              name: str,
              training_steps: int,
              network_architecture: typing.List[int],
              learning_rate: float,
              log_interval: int,
              batch_size: int,
              learning_starts: int,
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
    model.learn(total_timesteps=training_steps,
                log_interval=log_interval,
                tb_log_name=name,
                callback=custom_callback)
    model.save("models/%s" % name)


def QRDQN_train(environment: VecEnv,
                name: str,
                training_steps: int,
                network_architecture: typing.List[int],
                learning_rate: float,
                log_interval: int,
                batch_size: int,
                learning_starts: int,
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
    model.learn(total_timesteps=training_steps,
                log_interval=log_interval,
                tb_log_name=name,
                callback=custom_callback)
    model.save("models/%s" % name)


def HER_train(environment: VecEnv,
              name: str,
              training_steps: int,
              network_architecture: typing.List[int],
              learning_rate: float,
              log_interval: int,
              batch_size: int,
              learning_starts: int,
              **kwargs: typing.Any):
    model = HER("MlpPolicy",
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
    model.learn(total_timesteps=training_steps,
                log_interval=log_interval,
                tb_log_name=name,
                callback=custom_callback)
    model.save("models/%s" % name)


algorithms = {"DQN": DQN_train,
              "QRDQN": QRDQN_train,
              "HER": HER_train}
