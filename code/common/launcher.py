from common.utils import save_results, make_dir, save_result
from common.utils import plot_rewards, plot_price, plot_chargingnum
class Launcher:
    def __init__(self) -> None:
        pass
    def get_args(self):
        cfg = {}
        return cfg
    def env_agent_config(self,cfg):
        env,agent = None,None
        return env,agent
    def train(self,cfg, env, agent):
        res_dic = {}
        return res_dic

    def run(self):
        cfg = self.get_args()
        env, agent = self.env_agent_config(cfg)
        res_dic = self.train(cfg, env, agent)
        make_dir(plot_cfg.result_path, plot_cfg.model_path)
        # save_args(cfg,path = cfg['result_path']) # save parameters
        agent.save_model(path = cfg['model_path'])  # save models
        save_results(res_dic, tag = 'train', path = cfg['result_path']) # save results
        plot_rewards(res_dic['rewards'], cfg, path = cfg['result_path'],tag = "train")  # plot results
