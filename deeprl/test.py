import numpy as np
import gym
from experience_replay import ReplayDB
from experiment import Experiment
from agents import DQNAgent
import tflearn as nn


class TestAgent(object):
    def __init__(self, shape, n_actions):
        self.n_actions = n_actions
        self.db = ReplayDB(shape, 100)

    def select_action(self, obs):
        return np.random.choice(self.n_actions)

    def update(self, s, a, r, t):
        self.db.insert(s, a, r, t)


def create_mlp(inputs, n_out):
    net = nn.input_data(placeholder=inputs)
    net = nn.fully_connected(net, 25, activation='relu')
    net = nn.dropout(net, 0.4)
    net = nn.fully_connected(net, 25)
    net = nn.dropout(net, 0.4)
    net = nn.fully_connected(net, n_out, activation='linear')
    return net

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n
    agent = DQNAgent(create_mlp, n_actions, env.observation_space.shape)
    exp = Experiment(agent, env)
    exp.run_epoch(1000000)
    print agent.db.num_samples()
    print agent.db.sample(10)
