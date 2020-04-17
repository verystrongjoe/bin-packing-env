"""
https://squadrick.dev/journal/efficient-multi-gym-environments.html
"""

from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from environment import PalleteWorld
import multiprocessing

class MultiEnv:
    def __init__(self, num_env):
        self.envs = []
        for _ in range(num_env):
            self.envs.append(PalleteWorld(n_random_fixed=1))

    def reset(self):
        for env in self.envs:
            env.reset()

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        for env, ac in zip(self.envs, actions):
            ob, rew, done, info = env.step(ac)
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

            if done:
                env.reset()

        return obs, rewards, dones, infos


if __name__ == '__main__':
    envs = MultiEnv(num_env=10)