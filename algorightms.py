from stable_baselines3 import DQN, HER, PPO
import typing
from callback import CellworldCallback
from sb3_contrib import TRPO
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.ppo_recurrent import RecurrentPPO
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

def PPO_train(environment: VecEnv,
              name: str,
              training_steps: int,
              network_architecture: typing.List[int],
              learning_rate: float,
              log_interval: int,
              n_steps: int,
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


def RPPO_train(environment: VecEnv,
              name: str,
              training_steps: int,
              network_architecture: typing.List[int],
              learning_rate: float,
              log_interval: int,
              batch_size: int,
              n_steps: int,
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

def TRPO_train(environment: VecEnv,
               name: str,
               training_steps: int,
               network_architecture: typing.List[int],
               learning_rate: float,
               log_interval: int,
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


algorithms = {"DQN": DQN_train,
              "PPO": PPO_train,
              "QRDQN": QRDQN_train,
              "TRPO": TRPO_train,
              "RPPO": RPPO_train}
