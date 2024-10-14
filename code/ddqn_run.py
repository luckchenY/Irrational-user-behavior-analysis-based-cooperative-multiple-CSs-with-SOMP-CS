import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

import datetime
import argparse
import torch
from common.models import MLP
from common.memories import ReplayBufferQue
from DDQN.double_dqn import DoubleDQN
from common.launcher import Launcher
from envs.multi_env import EVenv
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # get current time
algo_name = "DoubleDQN"  # name of algorithm
env_name = 'Multi_env'  # name of environment

class DDQNConfig:
    ''' parameters of algorithm'''
    def __init__(self):
        self.algo_name = algo_name  # name pf algorithm
        self.env_name = env_name  # name of environment
        # self.device = cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
        self.train_eps = 1000  # episodes of training
        self.ep_max_steps = 168
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


    def env_agent_config(self,cfg, seed):
        ''' create env and agent
        '''  
        env = EVenv()
        env.seed(seed)
        n_states = env.state_dim  # dimension of state
        n_actions = env.action_space.shape[0]  # dimension of action
        print(f"n_states: {n_states}, n_actions: {n_actions}")
        cfg.update({"n_states":n_states,"n_actions":n_actions}) # update to cfg paramters
        models = {'Qnet':MLP(n_states,n_actions,hidden_dim=cfg['hidden_dim'])}
        memories = {'Memory':ReplayBufferQue(cfg['memory_capacity'])}
        agent = DoubleDQN(models,memories,cfg)
        return env,agent
        
    def train(self,cfg,env,agent):
        print("Start training!")
        print(f"Env: {cfg['env_name']}, Algorithm: {cfg['algo_name']}, Device: {cfg['device']}")
        rewards = []  # record rewards for all episodes
        ma_rewards = []
        for i_ep in range(cfg["train_eps"]):
            print("i_ep",i_ep)
            ep_reward = 0  # reward per episode
            ep_step = 0
            real_state, state = env.reset() # reset and obtain initial state
            # state=state[0]
            for t in range(cfg['ep_max_steps']):
                action = agent.sample_action(state) 
                print("action:",action)
                reward, real_state_, next_state, charging_num, EVLink_Price, action_input = env.step(t, action, real_state)
                ep_reward += reward
                if t == cfg['ep_max_steps'] - 1:
                    done = True
                agent.memory.push((state, action, reward, next_state, done)) 
                state = next_state 
                real_state = real_state_
                agent.update() 
                if done:
                    break
            if i_ep % cfg['target_update'] == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            rewards.append(ep_reward)
            if ma_rewards:
                ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
            else:
                ma_rewards.append(ep_reward)
            if (i_ep+1)%10 == 0: 
                print(f'Episode: {i_ep+1}/{cfg["train_eps"]}, Reward: {ep_reward:.2f}: Epislon: {agent.epsilon:.3f}')
        print("Finish training!")
        #env.close()
        res_dic = {'episodes':range(len(rewards)),'rewards':rewards,'ma_rewards':ma_rewards}
        return res_dic


if __name__ == "__main__":
    main = Main()
    main.run()
