from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class CellworldCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CellworldCallback, self).__init__(verbose)
        self.captures_in_episode = 0
        self.rewards = [0]
        self.captures = [0]
        self.captured = [0]
        self.survival = [0]
        self.finished = [0]
        self.truncated = [0]
        self.agents = {}
        self.stats_windows_size = 0
        self.current_survival = 0.0

    def _on_training_start(self):
        self.stats_windows_size = self.model._stats_window_size

    def _on_step(self):
        for env_id, info in enumerate(self.locals["infos"]):
            if 'terminal_observation' in info:
                self.rewards.append(info["reward"])
                if len(self.rewards) > self.stats_windows_size:
                    self.rewards.pop(0)

                self.captures.append(info["captures"])
                if len(self.captures) > self.stats_windows_size:
                    self.captures.pop(0)

                self.survival.append(info["survived"])
                if len(self.survival) > self.stats_windows_size:
                    self.survival.pop(0)

                self.captured.append(1 if info["captures"] > 0 else 0)
                if len(self.captured) > self.stats_windows_size:
                    self.captured.pop(0)

                self.finished.append(0 if info["TimeLimit.truncated"] else 1)
                if len(self.finished) > self.stats_windows_size:
                    self.finished.pop(0)

                self.truncated.append(1 if info["TimeLimit.truncated"] else 0)
                if len(self.truncated) > self.stats_windows_size:
                    self.truncated.pop(0)
                self.current_survival = safe_mean(self.survival)
                self.logger.record('cellworld/avg_captures', safe_mean(self.captures))
                self.logger.record('cellworld/survival_rate', self.current_survival)
                self.logger.record('cellworld/ep_finished', sum(self.finished))
                self.logger.record('cellworld/ep_truncated', sum(self.truncated))
                self.logger.record('cellworld/ep_captured', sum(self.captured))
                self.logger.record('cellworld/reward', safe_mean(self.rewards))

                for agent_name, agent_stats in info["agents"].items():
                    if agent_name not in self.agents:
                        self.agents[agent_name] = {}
                    for stat, value in agent_stats.items():
                        if stat not in self.agents[agent_name]:
                            self.agents[agent_name][stat] = []
                        self.agents[agent_name][stat].append(value)
                        if len(self.agents[agent_name][stat]) > self.stats_windows_size:
                            self.agents[agent_name][stat].pop(0)
                        stat_values = self.agents[agent_name][stat]
                        self.logger.record('cellworld/{}_{}'.format(agent_name, stat), safe_mean(stat_values))

        return True
