import gym
import numpy as np


class Experiment(object):
    ''' An RL experiment on given Gym environment

    Args:
        agent (object): RL agent. Should implement select_action and update
                        methods.
        env (str or object): either gym env object or environment name.
        phi (callable, optional): function to preprocess observations
    '''
    def __init__(self, agent, env, phi=None):
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env
        self.agent = agent
        self.phi = phi

    def run_episode(self, max_steps=0):
        ''' run a single episode

        Runs a single episode until termination or maximum number of steps is
        reached. On each step agent is called to select action and update.

        Args:
            max_steps (int): maximum number of episode transitions
                            (0 for no limit)

        Returns:
            (int) number of steps in episode
        '''
        obs = self.env.reset()
        done = False
        steps = 0
        reward = 0.
        while not done:
            a = self.agent.select_action(obs)
            obs_n, rew, done, _ = self.env.step(a)
            if self.phi:
                obs = self.phi(obs)
            self.agent.update(obs, a, rew, done)
            steps += 1
            obs = obs_n
            reward += rew
            if 0 < max_steps <= steps:
                break
        print "episode reward: %f" % reward
        return steps, reward

    def run_epoch(self, num_steps=50000):
        ''' Run environment for fixed number of steps (epoch)
        Args:
            num_steps (int): number of steps to run
        Returns:
            (int) number of episodes completed
        '''
        steps_left = num_steps
        num_eps = 0
        rewards = []
        while steps_left > 0:
            steps, rew = self.run_episode(steps_left)
            rewards.append(rew)
            steps_left -= steps
            num_eps += 1
        print "ran %i episodes, avg reward: %f" % (num_eps, np.mean(rewards))
        return rewards
