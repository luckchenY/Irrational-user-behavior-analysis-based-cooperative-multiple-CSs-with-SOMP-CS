import sys
import os
import torch
import datetime

from envs.environment import EVenv
from QLearning.QLearning import QLearning
from common.utils import plot_rewards
from common.utils import save_results, make_dir

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
algo_name = 'Q-learning'  # 算法名称
env_name = 'Q-Learning'  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EP_STEP = 168


class QlearningConfig:
    def __init__(self):
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = device  # 检测GPU
        self.train_eps = 1000  # 训练的回合数
        self.gamma = 0.9  # reward的衰减率
        self.epsilon_start = 0.65  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.001  # 学习率


class PlotConfig:
    ''' 绘图相关参数设置
    '''

    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = device  # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片


def env_agent_config(cfg, seed):
    # env = gym.make(cfg.env_name)
    # env = CliffWalkingWapper(env)
    evenv = EVenv()
    evenv.seed(seed)  # 设置随机种子
    state_dim = 48  # 状态维度
    action_dim = 81  # 动作维度
    agent = QLearning(state_dim, action_dim, cfg)
    return evenv, agent

def train(cfg, evenv, agent):
    print('开始训练！')
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    charging_nums = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个回合的奖励
        real_state, state = evenv.reset()  # 重置环境,即开始新的回合
        for t in range(MAX_EP_STEP):
            action = agent.choose_action(state)  # 根据算法选择一个动作
            # next_state, reward, done, _ = evenv.step(action)  # 与环境进行一次动作交互
            reward, real_state_, next_state, charging_num = evenv.step(t, action, real_state)
            done = False
            if t == MAX_EP_STEP - 1:
                done = True
            agent.update(state, action, reward, next_state, done)  # Q学习算法更新
            state = next_state
            real_state = real_state_
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('episode：{}/{}, cumulate reward：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('完成训练！')
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = QlearningConfig()
    plot_cfg = PlotConfig()
    # 训练
    mean_rewards = [0] * 1000
    mean_ma_rewards = [0] * 1000
    for s in range(20):
        print("training seed: ", s)
        evenv, agent = env_agent_config(cfg, seed=s)
        rewards, ma_rewards = train(cfg, evenv, agent)
        for i in range(len(rewards)):
            mean_rewards[i] = (mean_rewards[i] + rewards[i]) / 2
            mean_ma_rewards[i] = (mean_ma_rewards[i] + ma_rewards[i]) / 2
        print("mean_rewards: ", mean_rewards)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)
    agent.save(path=plot_cfg.model_path)
    save_results(mean_rewards, mean_ma_rewards, tag='train', path=plot_cfg.result_path)  # save results
    plot_rewards(mean_rewards, mean_ma_rewards, plot_cfg, tag="train")  # draw the figure of rewards
    '''
    evenv, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards = train(cfg, evenv, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save(path=plot_cfg.model_path)  # 保存模型
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果
    '''
    torch.cuda.empty_cache()
