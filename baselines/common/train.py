import os, gym, time, pickle, random, datetime
import torch, numpy as np
import ray, ray.exceptions
from tqdm import tqdm
from copy import deepcopy
from baselines.common.wrapper import NormalizedEnv
from baselines.common.plot import plot_train_result, plot_epoch_result
from baselines.common.buffer import SharedRolloutBuffer, SharedReplayBuffer, SharedPrioritizedReplayBuffer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_next_step(env, action):
    if isinstance(env.action_space, gym.spaces.Box):
        max_action = env.action_space.high
        min_action = env.action_space.low
        x = 0.5 * (action + 1) * (max_action - min_action) + min_action
        next_state, reward, terminated, info = env.step(x)
    else:
        next_state, reward, terminated, info = env.step(action)
    return next_state, reward, terminated, info


def evaluate(env, agent, seed, eval_iters, normalized_env=False):
    if normalized_env:
        env = NormalizedEnv(
            env=env, 
            obs_norm=True, 
            ret_norm=False,
            gamma=agent.gamma, 
            epsilon=agent.epsilon
            ) 
        
    total_ep_ret, total_ep_len = [], []

    for _ in tqdm(range(eval_iters), desc='EVALUATION'):
        ep_ret, ep_len, terminated = 0.0, 0, False
        rand_seed = seed + np.random.randint(0, 1000)
        state = env.reset(seed=rand_seed)

        while not terminated:
            action = agent.act(state, training=False)
            next_state, reward, terminated, _ = get_next_step(env, action)

            ep_ret += reward
            ep_len += 1
            state = next_state

        total_ep_ret.append(ep_ret)
        total_ep_len.append(ep_len)

    return total_ep_ret, total_ep_len








class Trainer:
    def __init__(self, env, eval_env, agent, seed):
        self.train_env = env
        self.eval_env = eval_env
        self.seed = seed
        self.agent = agent
        seed_all(self.seed)

    def train(self, 
            project_name, load_path, normalized_env, max_iters,
            eval_mode, eval_intervals, eval_iters, show_stats, show_graphs
            ):
        
        self.project_name = project_name
        self.load_path = load_path
        self.normalized_env = normalized_env
        self.max_iters = max_iters
        self.eval_mode = eval_mode
        self.eval_intervals = eval_intervals
        self.eval_iters = eval_iters

        self.show_stats = show_stats
        self.show_graphs = show_graphs
        self.epoch_logger = []
        self.train_logger = []
        self.start_time = time.time()
        self.start_time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        self.init_log = {
            'start_time': self.start_time_now,
            'env': self.train_env,
            'eval_env': self.eval_env,
            'agent': self.agent,
            'agent_config': self.agent.config,
            'seed': self.seed,
            'normalized_env': self.normalized_env,
            'project_name': self.project_name,
            'load_path': self.load_path,
            'max_iters': self.max_iters,
            'eval_mode': self.eval_mode,
            'eval_intervals': self.eval_intervals,
            'eval_iters': self.eval_iters
        }
        self.train_logger.append(self.init_log)

        print(f'============================+=====================================')
        print(f'TRAINING STARTING TIME      | {self.init_log["start_time"]}')
        print(f'PROJECT NAME                | {self.init_log["project_name"]}')
        print(f'ENVIRONMENT                 | {self.init_log["env"]}')
        print(f'EVALUATION ENVIRONMENT      | {self.init_log["eval_env"]}')
        print(f'AGENT                       | {self.init_log["agent"]}')
        print(f'SEED                        | {self.init_log["seed"]}')
        print(f'NORMALIZED ENVIRONMENT      | {self.init_log["normalized_env"]}')
        print(f'TOTAL TRAINING TIMESTEPS    | {self.init_log["max_iters"]}')
        print(f'EVALUATION MODE             | {self.init_log["eval_mode"]}')
        print(f'EVALUATION INTERVALS        | {self.init_log["eval_intervals"]}')
        print(f'EVALUATION ITERATIONS       | {self.init_log["eval_iters"]}')
        print(f'LOAD PATH                   | {self.init_log["load_path"]}')
        print(f'============================+=====================================')

        if load_path:
            self.agent.load(load_path)

        if normalized_env:
            self.train_env = NormalizedEnv(
                env=self.train_env, 
                obs_norm=True, 
                ret_norm=False,
                gamma=self.agent.gamma, 
                epsilon=self.agent.epsilon
                ) 

        global_stats = {
            'max_ret': -np.inf,
            'max_len': 0,
            'mean_ret': 0,
            'mean_len': 0,
            'ep_count': 0,
        }

        total_ep_ret, total_ep_len = [], []
        num_eps, ep_ret, ep_len = 0, 0, 0
        state, terminated = self.train_env.reset(seed=self.seed), False

        for timesteps in tqdm(range(self.max_iters), desc=f'TRAINNIG'):
            action = self.agent.act(state)
            next_state, reward, terminated, _ = get_next_step(self.train_env, action)
            result = self.agent.step(state, action, reward, next_state, terminated)

            ep_ret += reward
            ep_len += 1
            state = next_state

            if result is not None:
                self.epoch_logger.append({'timesteps': timesteps, 'result': result})

            if terminated:
                num_eps += 1
                state, terminated = self.train_env.reset(), False

                total_ep_ret.append(ep_ret)
                total_ep_len.append(ep_len)
                ep_ret, ep_len = 0, 0

            if timesteps % self.eval_intervals == 0:
                if self.eval_mode == True:
                    total_ep_ret, total_ep_len = evaluate(
                        self.eval_env, self.agent, self.seed, self.eval_iters, self.normalized_env)
                
                if total_ep_ret != [] and total_ep_len != []:
                    max_ep_ret = np.max(total_ep_ret)
                    max_ep_len = np.max(total_ep_len)
                    mean_ep_ret = np.mean(total_ep_ret)
                    mean_ep_len = np.mean(total_ep_len) 

                    self.epoch_logger.append({
                        'timesteps': timesteps,
                        'number_of_eps': num_eps,
                        'max_ep_ret': max_ep_ret,
                        'max_ep_len': max_ep_len,
                        'mean_ep_ret': mean_ep_ret,
                        'mean_ep_len': mean_ep_len,
                    })

                    global_stats['max_ret'] = max(global_stats['max_ret'], max_ep_ret)
                    global_stats['max_len'] = max(global_stats['max_len'], max_ep_len)
                    global_stats['mean_ret'] += mean_ep_ret
                    global_stats['mean_len'] += mean_ep_len
                    global_stats['ep_count'] += num_eps

                    if self.show_stats:
                        print(f'----------------------------+-------------------------------------')
                        print(f'TIMESTEPS                   | {timesteps}')   
                        print(f'THE NUMBER OF EPISODES      | {num_eps}')
                        print(f'MAX EPISODE LENGTH          | {max_ep_len}')
                        print(f'MAX EPISODE RETURN          | {round(max_ep_ret, 4)}')
                        print(f'MEAN EPISODE LENGTH         | {round(mean_ep_len, 4)}')
                        print(f'MEAN EPISODE RETURN         | {round(mean_ep_ret, 4)}')            
                        print(f'----------------------------+-------------------------------------')

                self.save_logs()
                num_eps = 0
                total_ep_ret, total_ep_len = [], []

        self.train_env.close()
        self.eval_env.close()

        self.end_time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.elapsed_time = datetime.timedelta(seconds=(time.time() - self.start_time))

        total_seconds = int(self.elapsed_time.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.time_elapse = f'{hours}h {minutes}m {seconds}s'

        global_stats["mean_len"] /= self.max_iters / self.eval_intervals
        global_stats["mean_ret"] /= self.max_iters / self.eval_intervals

        end_log = {
            'end_time': self.end_time_now,
            'tims_elapse': self.time_elapse,
            'total_episodes': global_stats["ep_count"],
            'global_max_len': global_stats["max_len"],
            'global_max_ret': global_stats["max_ret"],
            'global_mean_len': global_stats["mean_len"],
            'global_mean_ret': global_stats["mean_ret"],           
        }
        self.train_logger.append(end_log)
        
        print(f'============================+=====================================')
        print(f'TRINING FINISHING TIME      | {self.end_time_now}')
        print(f'TOTAL TRAINIMG ELAPSE       | {self.time_elapse}')
        print(f'TOTAL NUMBER OF EPISODES    | {global_stats["ep_count"]}')
        print(f'GLOBAL MAX EPISODE LENGTH   | {global_stats["max_len"]}')
        print(f'GLOBAL MAX EPISODE RETURN   | {round(global_stats["max_ret"], 4)}')
        print(f'GLOBAL MEAN EPISODE LENGTH  | {round(global_stats["mean_len"], 4)}')
        print(f'GLOBAL MEAN EPISODE RETURN  | {round(global_stats["mean_ret"], 4)}')
        print(f'============================+=====================================')

        self.save_logs()
        plot_train_result(project_name, self.epoch_logger, window=20, show_graphs=self.show_graphs)
        plot_epoch_result(project_name, self.epoch_logger, window=20, show_graphs=self.show_graphs)

    def get_logs(self):
        return self.train_logger, self.epoch_logger
    
    def save_logs(self):
        save_path = f'./log/{self.project_name}'
        os.makedirs(save_path, exist_ok=True)
        data_save_path = os.path.join(save_path, f'{self.project_name}.pkl')
        data = {'epoch_logger': self.epoch_logger, 'train_logger': self.train_logger}
        with open(data_save_path, 'wb') as f:
            pickle.dump(data, f)








class DistributedTrainer:
    def __init__(self, env, eval_env, agent, seed):
        self.train_env = env
        self.eval_env = eval_env
        self.learner = agent
        self.seed = seed 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.environ['RAY_memory_monitor_refresh_ms'] = '0'
        os.environ['RAY_verbose_spill_logs'] = '0'
        os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'
        seed_all(seed)

    def train(self, 
            project_name, load_path, normalized_env, max_iters, 
            n_runners, runner_iters, eval_mode, eval_intervals, eval_iters, 
            policy_type, action_type, show_stats, show_graphs
            ):
        
        self.project_name = project_name
        self.load_path = load_path
        self.normalized_env = normalized_env
        self.max_iters = max_iters
        self.n_runners = n_runners
        self.runner_iters = runner_iters
        self.eval_mode = eval_mode
        self.eval_intervals = eval_intervals
        self.eval_iters = eval_iters

        self.policy_type = policy_type
        self.action_type = action_type
        self.show_stats = show_stats
        self.show_graphs = show_graphs
        self.epoch_logger = []
        self.train_logger = []
        self.start_time = time.time()
        self.start_time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.state_dim = self.train_env.observation_space.shape[0]
        if self.action_type == 'continuous':
            self.action_dim = self.train_env.action_space.shape[0]
        elif self.action_type == 'discrete':
            self.action_dim = self.train_env.action_space.n
        elif self.action_type == 'multidiscrete':
            self.action_dim = self.train_env.action_space.nvec 

        if self.policy_type == 'off_policy':
            self.prioritized_mode = self.learner.prioritized_mode
            self.batch_size = self.learner.batch_size            
            self.prio_alpha = self.learner.prio_alpha
            self.prio_beta = self.learner.prio_beta 
            self.prio_eps = self.learner.prio_eps

        self.init_log = {
            'start_time': self.start_time_now,
            'env': self.train_env,
            'eval_env': self.eval_env,
            'agent': self.learner,
            'agent_config': self.learner.config,
            'seed': self.seed,
            'normalized_env': self.normalized_env,
            'project_name': self.project_name,
            'load_path': self.load_path,
            'max_iters': self.max_iters,
            'n_runners': self.n_runners,
            'runner_iters': self.runner_iters,
            'eval_mode': self.eval_mode,
            'eval_intervals': self.eval_intervals,
            'eval_iters': self.eval_iters
        }

        print(f'============================+=====================================')
        print(f'TRAINING STARTING TIME      | {self.init_log["start_time"]}')
        print(f'PROJECT NAME                | {self.init_log["project_name"]}')
        print(f'ENVIRONMENT                 | {self.init_log["env"]}')
        print(f'EVALUATION ENVIRONMENT      | {self.init_log["eval_env"]}')
        print(f'AGENT                       | {self.init_log["agent"]}')
        print(f'SEED                        | {self.init_log["seed"]}')
        print(f'NORMALIZED ENVIRONMENT      | {self.init_log["normalized_env"]}')
        print(f'TOTAL TRAINING TIMESTEPS    | {self.init_log["max_iters"]}')
        print(f'THE NUMBER OF RUNNERS       | {self.init_log["n_runners"]}')
        print(f'ITERATIONS PER A RUNNER     | {self.init_log["runner_iters"]}')        
        print(f'EVALUATION MODE             | {self.init_log["eval_mode"]}')
        print(f'EVALUATION INTERVALS        | {self.init_log["eval_intervals"]}')
        print(f'EVALUATION ITERATIONS       | {self.init_log["eval_iters"]}')
        print(f'LOAD PATH                   | {self.init_log["load_path"]}')
        print(f'============================+=====================================')

        self.train_logger.append(self.init_log)
        path = f'./log/{self.project_name}'
        os.makedirs(path, exist_ok=True)
        path = path + '/DistributedAgent.pth'

        if load_path:
            self.learner.load(load_path)
            
        ray.init()    
        try:
            self.buffer_size = self.learner.buffer_size
            self.reward_norm = self.learner.reward_norm
            self.epsilon = 1e-8
        
            if self.policy_type == 'on_policy':
                buffer = SharedRolloutBuffer.remote(
                    state_dim=self.state_dim, 
                    action_dim=self.action_dim, 
                    buffer_size=self.buffer_size, 
                    device=self.device, 
                    reward_norm=self.reward_norm, 
                    epsilon=self.epsilon
                    )
                
            elif self.policy_type == 'off_policy':
                if self.prioritized_mode:
                    buffer = SharedPrioritizedReplayBuffer.remote(
                        state_dim=self.state_dim, 
                        action_dim=self.action_dim, 
                        buffer_size=self.buffer_size, 
                        batch_size=self.batch_size,
                        device=self.device, 
                        alpha=self.prio_alpha,
                        reward_norm=self.reward_norm, 
                        epsilon=self.epsilon
                        )
                else:
                    buffer = SharedReplayBuffer.remote(
                        state_dim=self.state_dim, 
                        action_dim=self.action_dim, 
                        buffer_size=self.buffer_size, 
                        batch_size=self.batch_size,
                        device=self.device, 
                        reward_norm=self.reward_norm, 
                        epsilon=self.epsilon
                        )

            runners = [
                Runner.remote(
                    name=runner_name, 
                    env=self.train_env, 
                    learner=self.learner, 
                    seed=self.seed, 
                    max_iters=self.max_iters,
                    policy_type=self.policy_type,
                    load_path=path, 
                    normalized_env=self.normalized_env) 
                    for runner_name in range(self.n_runners)
                    ]
            
            runner_tasks = []
            self.learner.save(path)
            timesteps, num_eps, train_iters = 0, 0, 0
            eval_counter = 0

            while timesteps < max_iters:
                train_iters = 0
                for runner in runners:
                    runner_tasks.append(
                        runner.run.remote(
                            buffer=buffer, 
                            runner_iters=runner_iters
                        ))
                    time.sleep(0.1)

                results = ray.get(runner_tasks)
                runner_tasks.clear()

                total_ep_ret, total_ep_len = [], []
                for result in results:
                    name, time_per_run, ep_per_run, ep_ret, ep_len, elapse = result

                    num_eps += ep_per_run
                    train_iters += time_per_run   

                    total_ep_ret += ep_ret
                    total_ep_len += ep_len

                    print(f"RUNNER {name} | TOTAL EPISODES: {ep_per_run}, TOTAL TIMESTEPS: {time_per_run}, ELAPSE: {elapse[1]}m {elapse[2]}s")

                eval_counter += train_iters
                buffer_size = ray.get(buffer.size.remote())
                if buffer_size >= self.learner.update_after:
                    print('\n===================== LEARNER STARTS TRAININGS ====================\n') 
                    result = None
                    if self.policy_type == 'on_policy':
                        states, actions, rewards, next_states, dones = ray.get(buffer.sample.remote())
                        result = self.learner.learn(states, actions, rewards, next_states, dones)

                        if result is not None:
                            self.epoch_logger.append({'timesteps': timesteps, 'result': result})

                    elif self.policy_type == 'off_policy':
                        for t in tqdm(range(1, train_iters+1), desc=f'TRAINING'):
                            if self.prioritized_mode:
                                fraction = min((timesteps + t) / max_iters, 1.)
                                self.prio_beta = self.prio_beta + fraction * (1. - self.prio_beta)

                                states, actions, rewards, next_states, dones, weights, idxs = ray.get(buffer.sample.remote(self.prio_beta))
                                result = self.learner.learn(states, actions, rewards, next_states, dones, weights, timesteps + t)

                                if result['td_error'] is not None:
                                    td_error = result['td_error'].detach().cpu().abs().numpy().flatten()
                                    new_prios = td_error + self.prio_eps
                                    buffer.update_priorities.remote(idxs, new_prios)
                            else:
                                states, actions, rewards, next_states, dones = ray.get(buffer.sample.remote())
                                result = self.learner.learn(states, actions, rewards, next_states, dones, 
                                                            weights=None, global_timesteps=timesteps + t)
                            
                            if result is not None:
                                self.epoch_logger.append({'timesteps': timesteps + t, 'result': result})
                    
                    self.learner.save(path)
                else:
                    print('\n================= LEARNER IS WAITING FOR TRAININGS ================\n')
                
                timesteps += train_iters     
                if eval_counter >= self.eval_intervals:
                    eval_counter -= self.eval_intervals
                    if self.eval_mode == True:
                        total_ep_ret, total_ep_len = evaluate(
                            self.eval_env, self.learner, self.seed, self.eval_iters, self.normalized_env)

                    if total_ep_ret != [] and total_ep_len != []:
                        max_ep_ret = np.max(total_ep_ret)
                        max_ep_len = np.max(total_ep_len)
                        mean_ep_ret = np.mean(total_ep_ret)
                        mean_ep_len = np.mean(total_ep_len) 

                        self.epoch_logger.append({
                            'timesteps': timesteps,
                            'number_of_eps': num_eps,
                            'max_ep_ret': max_ep_ret,
                            'max_ep_len': max_ep_len,
                            'mean_ep_ret': mean_ep_ret,
                            'mean_ep_len': mean_ep_len,
                        })

                        if self.show_stats:
                            print(f'----------------------------+-------------------------------------')
                            print(f'TIMESTEPS                   | {timesteps}')   
                            print(f'THE NUMBER OF EPISODES      | {num_eps}')
                            print(f'MAX EPISODE LENGTH          | {max_ep_len}')
                            print(f'MAX EPISODE RETURN          | {round(max_ep_ret, 4)}')
                            print(f'MEAN EPISODE LENGTH         | {round(mean_ep_len, 4)}')
                            print(f'MEAN EPISODE RETURN         | {round(mean_ep_ret, 4)}')            
                            print(f'----------------------------+-------------------------------------')

                self.save_logs()              
                
        except KeyboardInterrupt:        
            print('\n================== TRAINING HAS BEEN SHUTDOWN =====================\n')

        finally:
            self.train_env.close()
            self.eval_env.close()
            ray.shutdown()

            self.end_time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.elapsed_time = datetime.timedelta(seconds=(time.time() - self.start_time))

            total_seconds = int(self.elapsed_time.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_elapse = f'{hours}h {minutes}m {seconds}s'

            end_log = {
                'end_time': self.end_time_now,
                'tims_elapse': self.time_elapse,
            }
            self.train_logger.append(end_log)
        
            print(f'============================+=================================')
            print(f'TRINING FINISHING TIME      | {self.end_time_now}')
            print(f'TOTAL TRAINIMG ELAPSE       | {self.time_elapse}')
            print(f'============================+=================================')

            self.save_logs()
            plot_train_result(project_name, self.epoch_logger, window=20, show_graphs=self.show_graphs)
            plot_epoch_result(project_name, self.epoch_logger, window=20, show_graphs=self.show_graphs)

    def get_logger(self):
        return self.train_logger, self.epoch_logger
    
    def save_logs(self):
        save_path = f'./log/{self.project_name}'
        os.makedirs(save_path, exist_ok=True)
        data = {'train_logger': self.train_logger, 'epoch_logger': self.epoch_logger}
        loggers_save_path = os.path.join(save_path, f'{self.project_name}.pkl')
        with open(loggers_save_path, 'wb') as f:
            pickle.dump(data, f)


@ray.remote(num_gpus=0.1)
class Runner:
    def __init__(self, name, env, learner, seed, max_iters, policy_type, load_path, normalized_env):
        self.name = name
        self.env = deepcopy(env)
        if normalized_env:
            self.env = NormalizedEnv(
                env=self.env,
                obs_norm=True, 
                ret_norm=False,
                gamma=learner.gamma, 
                epsilon=learner.epsilon
                ) 

        self.runner = deepcopy(learner)
        self.seed = seed + 100 * name
        self.max_iters = max_iters
        self.policy_type = policy_type
        self.load_path = load_path

        self.state = self.env.reset(seed=self.seed)
        self.ep_ret, self.ep_len = 0, 0

    def run(self, buffer, runner_iters):
        ep_counter, timesteps = 0, 0
        total_ep_ret, total_ep_len = [], []
        self.runner.load(self.load_path)

        start_time = time.time()
        while timesteps < runner_iters:
            buffer_size = ray.get(buffer.size.remote())
            if self.policy_type == 'on_policy':
                if buffer_size >= self.runner.update_after:
                    break
            elif self.policy_type == 'off_policy':
                if buffer_size >= self.max_iters:
                    break

            timesteps += 1
            action = self.runner.act(self.state, buffer_size)
            next_state, reward, terminated, _ =  get_next_step(self.env, action)
            buffer.store.remote(self.state, action, reward, next_state, terminated)

            self.ep_ret += reward
            self.ep_len += 1
            self.state = next_state 

            if terminated:
                ep_counter += 1
                self.state, terminated = self.env.reset(), False

                total_ep_ret.append(self.ep_ret)
                total_ep_len.append(self.ep_len)
                self.ep_ret, self.ep_len = 0, 0

        elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
        total_seconds = int(elapsed_time.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapse = [hours, minutes, seconds]

        return self.name, timesteps, ep_counter, total_ep_ret, total_ep_len, elapse