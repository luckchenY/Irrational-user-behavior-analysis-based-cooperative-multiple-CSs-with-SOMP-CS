import torch
import datetime
from common.utils import save_results, make_dir
from common.utils import plot_rewards
from DQN.dqn import DQN
from envs.individual_environment import EVenv

cuda = torch.device("cuda:" + str(0))
curr_path = "/home/pc/LIUJIE/RL-Pricing-Charging-Schedule/code"
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # get current time
algo_name = "DQN"  # name of algorithm
env_name = 'Individual_Station'  # name of environment

MAX_EP_STEP = 168

class DQNConfig:
    ''' parameters of algorithm'''

    def __init__(self):
        self.algo_name = algo_name  # name pf algorithm
        self.env_name = env_name  # name of environment
        self.device = cuda
        self.train_eps = 3000  # episodes of training
        # super-parameters
        self.gamma = 0.95  # discount factor \gamma
        self.epsilon_start = 0.90  # start-epsilon in e-greedy
        self.epsilon_end = 0.01  # end-epsilon in e-greedy
        self.epsilon_decay = 500  # decay rate of epsilon
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


def env_agent_config(cfg, seed=1):
    ''' create environment and agent'''
    evenv = EVenv()   # create environment
    evenv.seed(seed)  # set seed
    EVLink_Price = evenv.EVLink_price
    state_dim = evenv.state_dim  # dimension of state
    action_dim = evenv.action_space.shape[0]  # dimension of action
    agent = DQN(state_dim, action_dim, cfg)  # create agent
    return EVLink_Price, evenv, agent

def train(cfg, evenv, agent):
    ''' 训练
    '''
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # record all rewards in each episode
    ma_rewards = []  # record all slipping rewards in each episode
    charging_nums = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # record the cumulative reward in one episode
        # ep_charging_num = 0  # num of EVs charging in an ep
        real_state, state = evenv.reset()  # reset environment
        for t in range(MAX_EP_STEP):
            action = agent.choose_action(state)  # choose action
            reward, real_state_, next_state, charging_num = evenv.step(t, action, real_state)  # return transition
            charging_nums.append(charging_num)
            done = False
            if t == MAX_EP_STEP - 1:
                done = True
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state  # update next state
            real_state = real_state_
            agent.update()  # update agent
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
    print('Training completed！')
    return rewards, ma_rewards, charging_nums

if __name__ == "__main__":
    cfg = DQNConfig()
    plot_cfg = PlotConfig()
    # 训练
    EVLink_Price, evenv, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards, charging_nums = train(cfg, evenv, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)  # create directory for saving models and results
    agent.save(path=plot_cfg.model_path)  # save model
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)  # save results
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")     # draw the figure of rewards
    torch.cuda.empty_cache()