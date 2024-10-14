import sys,os
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

import datetime
import gym
import torch
import argparse

# from env import NormalizedActions,OUNoise
from envs.multi_env import EVenv
from DDPG.ddpg import DDPG
from common.utils import save_results,make_dir
from common.utils import plot_rewards,save_args

def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DDPG',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='Pendulum-v1',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=100,type=int,help="episodes of training")
    parser.add_argument('--inter_eps',default=100,type=int,help="episodes of training")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--critic_lr',default=1e-3,type=float,help="learning rate of critic")
    parser.add_argument('--actor_lr',default=1e-4,type=float,help="learning rate of actor")
    parser.add_argument('--memory_capacity',default=8000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--target_update',default=2,type=int)
    parser.add_argument('--soft_tau',default=1e-2,type=float)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/' )
    parser.add_argument('--model_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/models/' ) # path to save models
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")   
    args = parser.parse_args()                           
    return args

def env_agent_config(cfg, seed):
    env = NormalizedActions(gym.make(cfg.env_name)) # 装饰action噪声
    env.seed(seed) # 随机种子
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = DDPG(n_states,n_actions,cfg)
    return env,agent
def train(cfg, env, agent):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    ou_noise = OUNoise(env.action_space)  # noise of action
    rewards = [] # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        print("i_ep:", i_ep)
        state = env.reset()
        #print("reset0: ", state)
        state = state[0]
        #print("reset1: ", state)
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        for ep_max in range(cfg.inter_eps):
            i_step += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_step)
            #print("action:",action) 
            next_state, reward, done, _, _ = env.step(action)
            #print("next_state:",next_state)
            ep_reward += reward
            #print("ep_reward:",ep_reward)
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            ep_max +=1
        if (i_ep+1)%10 == 0:
            print(f'Env:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.2f}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Finish training!')
    return {'rewards':rewards,'ma_rewards':ma_rewards}

if __name__ == "__main__":
    cfg = get_args()
    # training
    env,agent = env_agent_config(cfg,seed=1)
    res_dic = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    #save_args(cfg)
    agent.save(path=cfg.model_path)
    #save_results(res_dic, tag='train', path=cfg.result_path)  
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train")  
