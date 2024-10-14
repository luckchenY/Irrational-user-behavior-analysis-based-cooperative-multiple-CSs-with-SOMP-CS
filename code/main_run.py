from ctypes.wintypes import tagRECT
import os
import sys
import numpy as np
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current directory
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add path to system path

import torch
import datetime
from common.utils import save_results, make_dir, save_result
from common.utils import plot_rewards, plot_price, plot_chargingnum
from DQN.dqn import DQN
from envs.environment import EVenv
import matplotlib.pyplot as plt

# cuda = torch.device("cuda:" + str(1))
# curr_path = "/home/pc/LIUJIE/RL-Pricing-Charging-Schedule/code"
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # get current time
algo_name = "DQN"  # name of algorithm
env_name = 'EV_Charging_Pricing'  # name of environment

MAX_EP_STEP = 168

class DQNConfig:
    ''' parameters of algorithm'''

    def __init__(self):
        self.algo_name = algo_name  # name pf algorithm
        self.env_name = env_name  # name of environment
        # self.device = cuda
        self.device = torch.device(
           "cuda" if torch.cuda.is_available() else "cpu")  # check GPU
        self.train_eps = 1000  # episodes of training
        # super-parameters
        self.gamma = 0.95  # discount factor \gamma
        self.epsilon_start = 0.90  # start-epsilon in e-greedy
        self.epsilon_end = 0.01  # end-epsilon in e-greedy
        self.epsilon_decay = 500  # decay rate of epsilon
        # self.epsilon_start = 0.6
        # self.epsilon_end = 0.01
        # self.epsilon_decay = 500
        self.lr = 0.0001  # learning rate
        self.memory_capacity = 100000  # capacity of memory buffer
        self.batch_size = 64  # the size of mini-batch in SGD
        self.target_update = 4  # update rate of target network
        self.hidden_dim = 256  # hidden layer of network


class PlotConfig:
    ''' parameters of drawing'''
    def __init__(self) -> None:
        self.algo_name = algo_name  # name of algorithm
        self.env_name = env_name  # name of environment
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check gpu
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # path to save models
        self.save = True  # whether save picture


def env_agent_config(cfg, seed):
    ''' create environment and agent'''
    evenv = EVenv()   # create environment
    evenv.seed(seed)  # set seed
    EVLink_Price = evenv.EVLink_price
    state_dim = evenv.state_dim  # dimension of state
    action_dim = evenv.action_space.shape[0]  # dimension of action
    agent = DQN(state_dim, action_dim, cfg)  # create agent
    return EVLink_Price, evenv, agent

def train(cfg, evenv, agent):
    ''' Train
    '''
    print('start to train!')
    print(f'Environment:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device: {cfg.device}')
    rewards = []  # record all rewards in each episode
    ma_rewards = []  # record all slipping rewards in each episode
    charging_nums = []

    EVLink_Prices=[]
    prices1=[]
    prices2=[]
    prices3=[]
    prices4=[]
    rates1=[]
    rates2=[]
    rates3=[]
    rates4=[]

    gradient = torch.zeros(256)
    grad_output = torch.zeros(256)

    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # record the cumulative reward in one episode
        # ep_charging_num = 0  # num of EVs charging in an ep
        real_state, state = evenv.reset()  # reset environment
        for t in range(MAX_EP_STEP):
            action = agent.choose_action(state)  # choose action
            reward, real_state_, next_state, charging_num, EVLink_Price, action_input = evenv.step(t, action, real_state)  # return transition
            EVLink_Prices.append(EVLink_Price)
            # print("time: ", t, "reward: ", reward)
            # print("time: ", t, "real_state_: ", real_state_)
            # if i_ep==599:
            #     if t>=118 and t<168:
            #         EVLink_Prices.append(EVLink_Price)
            #         prices1.append(action_input[0])
            #         prices2.append(action_input[1])
            #         prices3.append(action_input[2])
            #         prices4.append(action_input[3])
            #         rates1.append(action_input[4])
            #         rates2.append(action_input[5])
            #         rates3.append(action_input[6])
            #         rates4.append(action_input[7])
            charging_nums.append(charging_num)
            done = False
            if t == MAX_EP_STEP - 1:
                done = True
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state  # update next state
            real_state = real_state_
            gradient = agent.update()  # update agent
            ep_reward += reward  # cumulate reward
            # ep_charging_num += charging_num
            if done:
                break
        if (i_ep+1) % cfg.target_update == 0:  # update target network
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        # charging_nums.append(ep_charging_num)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1) % 10 == 0:
            print('episode：{}/{}, cumulate reward：{}'.format(i_ep+1, cfg.train_eps, ep_reward))
        # if i_ep == 990:
        #     grad_output = gradient
    print('Training completed！')
    # return rewards, ma_rewards, charging_nums, grad_output, EVLink_Prices, prices1, prices2, rates1, rates2
    return rewards, ma_rewards, charging_nums, EVLink_Prices



if __name__ == "__main__":
    cfg = DQNConfig()
    plot_cfg = PlotConfig()
    # 训练
    EVLink_Price, evenv, agent = env_agent_config(cfg, seed=1)
    # rewards, ma_rewards, charging_nums, grad_output, EVLink_Prices, prices1, prices2, rates1, rates2 = train(cfg, evenv, agent)
    rewards, ma_rewards, charging_nums, EVLink_Prices = train(cfg, evenv, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)  # create directory for saving models and results
    agent.save(path=plot_cfg.model_path)  # save model
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)  # save results
    # save_result(grad_output, tag='gradient', path=plot_cfg.result_path)

    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # draw the figure of rewards

    # save_result(EVLink_Prices, tag='EVLink_Prices', path=plot_cfg.result_path)
    # save_result(prices1, tag='prices1', path=plot_cfg.result_path)
    # save_result(prices2, tag='prices2', path=plot_cfg.result_path)
    # save_result(rates1, tag='rates1', path=plot_cfg.result_path)
    # save_result(rates2, tag='rates2', path=plot_cfg.result_path)
    # plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.xlabel('Episodes')       # 回合数
    # plt.plot(EVLink_Prices, label='EVLink_Price')
    # plt.plot(prices1, label='price1')
    # plt.plot(prices2, label='price2')
    # plt.plot(rates1, label='rate1')
    # plt.plot(rates2, label='rate2')
    # plt.legend()
    # plt.savefig(plot_cfg.result_path+"example")
    # plt.show()


    save_result(charging_nums, tag='# of EVs', path=plot_cfg.result_path)
    plot_chargingnum(charging_nums, plot_cfg)
    save_result(EVLink_Prices, tag='EVLink', path=plot_cfg.result_path)
    plot_price(EVLink_Price, plot_cfg, tag='EVLink')
    torch.cuda.empty_cache()
    sys.exit()
