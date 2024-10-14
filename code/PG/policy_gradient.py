import torch
from torch.distributions import Categorical
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    ''' 多层感知机
        输入：state维度
        输出：概率
    '''

    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        # 24和36为hidden layer的层数，可根据input_dim, action_dim的情况来改变
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 81)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        x = F.softmax(self.fc2(x), dim=-1)
        # print("x: ", x)
        return x

class PolicyGradient:

    def __init__(self, state_dim, cfg):
        self.gamma = cfg.gamma
        # self.policy_net = MLP(state_dim, hidden_dim=cfg.hidden_dim)
        self.policy_net = MLP(5, 81)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr)
        self.batch_size = cfg.batch_size

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        # print("state: ", state)
        probs = self.policy_net(state)
        # m = Bernoulli(probs)
        m = Categorical(probs)
        action = m.sample()
        # action = action.data.numpy().tolist()
        action = action.item()
        # print("action: ", action)
        return action

    def update(self, reward_pool, state_pool, action_pool):
        gradient = torch.zeros(128)
        # Discount reward
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # Gradient Descent
        self.optimizer.zero_grad()

        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            # m = Bernoulli(probs)
            m = Categorical(probs)
            loss = -m.log_prob(action) * reward   # negative score function x reward
            # print("loss: ", loss)
            loss.backward()
            for param in self.policy_net.parameters():
                gradient = param.grad
        self.optimizer.step()
        return gradient

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'pg_checkpoint.pt')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'pg_checkpoint.pt'))
