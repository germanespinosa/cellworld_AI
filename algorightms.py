from stable_baselines3 import DQN, PPO
import typing
from sb3_contrib import TRPO
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from collections import namedtuple

Algorithm = namedtuple("algorithm", ['create', 'load'])


def DQN_create(environment: VecEnv,
               training_steps: int,
               network_architecture: typing.List[int],
               learning_rate: float,
               batch_size: int,
               learning_starts: int,
               tensorboard_log: str,
               **kwargs: typing.Any):
    return DQN("MlpPolicy",
               environment,
               verbose=1,
               batch_size=batch_size,
               learning_rate=learning_rate,
               train_freq=(1, "step"),
               buffer_size=training_steps,
               learning_starts=learning_starts,
               replay_buffer_class=ReplayBuffer,
               policy_kwargs={"net_arch": network_architecture},
               tensorboard_log=tensorboard_log
               )


def QRDQN_create(environment: VecEnv,
                 training_steps: int,
                 network_architecture: typing.List[int],
                 learning_rate: float,
                 batch_size: int,
                 learning_starts: int,
                 tensorboard_log: str,
                 **kwargs: typing.Any):
    return QRDQN("MlpPolicy",
                 environment,
                 verbose=1,
                 batch_size=batch_size,
                 learning_rate=learning_rate,
                 train_freq=(1, "step"),
                 buffer_size=training_steps,
                 learning_starts=learning_starts,
                 replay_buffer_class=ReplayBuffer,
                 policy_kwargs={"net_arch": network_architecture},
                 tensorboard_log=tensorboard_log
    )


def PPO_create(environment: VecEnv,
               network_architecture: typing.List[int],
               learning_rate: float,
               n_steps: int,
               tensorboard_log: str,
               **kwargs: typing.Any):
    return PPO("MlpPolicy",
               environment,
               learning_rate=learning_rate,
               n_steps=n_steps,
               policy_kwargs={"net_arch": network_architecture},
               verbose=1,
               tensorboard_log=tensorboard_log)


def RPPO_create(environment: VecEnv,
                network_architecture: typing.List[int],
                learning_rate: float,
                batch_size: int,
                n_steps: int,
                tensorboard_log: str,
                **kwargs: typing.Any):
    return RecurrentPPO("MlpLstmPolicy",
                        environment,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        policy_kwargs={"net_arch": network_architecture},
                        n_steps=n_steps,
                        verbose=1,
                        tensorboard_log=tensorboard_log)


def TRPO_create(environment: VecEnv,
                network_architecture: typing.List[int],
                learning_rate: float,
                tensorboard_log: str,
                **kwargs: typing.Any):
    return TRPO("MlpPolicy",
                environment,
                verbose=1,
                learning_rate=learning_rate,
                policy_kwargs={"net_arch": network_architecture},
                tensorboard_log=tensorboard_log)


algorithms = {"DQN": Algorithm(DQN_create, DQN.load),
              "PPO": Algorithm(PPO_create, PPO.load),
              "QRDQN": Algorithm(QRDQN_create, QRDQN.load),
              "TRPO": Algorithm(TRPO_create, TRPO.load),
              "RPPO": Algorithm(RPPO_create, RecurrentPPO.load)}
