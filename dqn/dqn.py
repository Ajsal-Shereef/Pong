import time
import numpy as np
import torch
import random
import torch.optim as optim
import torchinfo
from tqdm import tqdm
from torchsummary import summary
from scipy.special import softmax
from scipy.stats import entropy

from copy import deepcopy
from human_oracle.oracle import HumanOracle
from learning_agent.architectures.mlp import MLP
from torch.nn.utils import clip_grad_norm_
from learning_agent.priortized_replay_buffer import PrioritizedReplayBuffer
from learning_agent.replay_buffer import ReplayBuffer
from dqn.dqn_utils import calculate_dqn_loss
from learning_agent import common_utils
from learning_agent.agent import Agent
from human_oracle.oracle import HumanOracle
from reply_buffer.lstm_buffer_handler import LSTMBufferHandler
from utils.utils import *
from lstm.lstm_training_handler import LSTMTrainingHandler
from preference_learning.training_handler import TrainingHandler
from utils.plot_network_internals import PlotInternal
from scipy.special import kl_div
from env.env import Env

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """
    Attribute:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (PrioritizedReplayBuffer): replay memory
        dqn (nn.Module): actor model to select actions
        dqn_target (nn.Module): target actor model to select actions
        dqn_optim (Optimizer): optimizer for training actor
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step number
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        epsilon (float): parameter for epsilon greedy controller_policy
        n_step_buffer (deque): n-size buffer to calculate n-step returns
        per_beta (float): beta parameter for prioritized replay buffer
        use_n_step (bool): whether or not to use n-step returns
    """

    def __init__(self, env, args, log_cfg, hyper_params, network_cfg, optim_cfg, dump_dir, lstm_config, logger):
        """Initialize."""
        Agent.__init__(self, env, args, log_cfg)

        self.episode_step = 0
        self.total_step = 0
        self.i_episode = 0
        self.logger = logger
        self.dump_dir, self.lstm_config = dump_dir, lstm_config
        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        self.per_beta = hyper_params.per_beta
        self.use_n_step = hyper_params.n_step > 1
        self.use_prioritized = hyper_params.use_prioritized


        self.max_epsilon = hyper_params.max_epsilon
        self.min_epsilon = hyper_params.min_epsilon
        self.epsilon = hyper_params.max_epsilon

        self._initialize()
        self._init_network()
        self.reset_frame_array()
        self.set_log_dictionary()
        self.is_write_video = False
        
        #summary(self.dqn_target.cuda(), (1,74,74))

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        # replay memory for a single step
        if self.use_prioritized:
            self.memory = PrioritizedReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                alpha=self.hyper_params.per_alpha,
            )
        # use ordinary replay buffer
        else:
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size,
                batch_size=self.hyper_params.batch_size,
                gamma=self.hyper_params.gamma,
            )

        # replay memory for multi-steps
        if self.use_n_step:
            self.memory_n = ReplayBuffer(
                self.hyper_params.buffer_size,
                batch_size=self.hyper_params.batch_size,
                n_step=self.hyper_params.n_step,
                gamma=self.hyper_params.gamma,
            )

    def set_log_dictionary(self):
        self.log_dict = ["Episode" , "DQN loss", "Avg loss", "Epsilon", "Score", "Episode steps"]
        if self.args.mode == 'preference':
            self.personalized_policy_log_dict = ["Personalized policy Episodes", "personalized policy Score", "Personalised_policy_Cummulative_preferred_region", 
                                                 "Personalised_policy_episode_preferred_region", "DQN policy Score", "DQN_policy_Cummulative_preferred_region",
                                                 "DQN_policy_episode_preferred_region"]
        elif self.args.mode == 'avoid':
            self.personalized_policy_log_dict = ["Personalized policy Episodes", "personalized policy Score", "Personalised_policy_Cummulative_avoid_region", 
                                                 "Personalised_policy_episode_avoid_region", "DQN policy Score", "DQN_policy_Cummulative_avoid_region",
                                                 "DQN_policy_episode_avoid_region"]
        else:
            self.personalized_policy_log_dict = ["Personalized policy Episodes", "personalized policy Score", "Personalised_policy_Cummulative_preferred_region", 
                                                 "Personalised_policy_Cummulative_avoid_region", "Personlaised_policy_episode_preferred_region", 
                                                 "Personalised_policy_episode_avoid_region", "DQN policy Score", "DQN_policy_Cummulative_preferred_region", 
                                                 "DQN_policy_Cummulative_avoid_region", "DQN_policy_episode_preferred_region", "DQN_policy_episode_avoid_region"]
    
    def _init_network(self):
        """Initialize networks and optimizers."""
        self.dqn = MLP(input_size=self.network_cfg.fc_input_size,
                       output_size=self.args.num_action,
                       hidden_sizes = [256,512,512]).to(device)
        # if torch.cuda.device_count() > 1:
        #     self.dqn = torch.nn.DataParallel(self.dqn)
        self.dqn_target = MLP(input_size=self.network_cfg.fc_input_size,
                              output_size=self.args.num_action,
                              hidden_sizes = [256,512,512]).to(device)
        # if torch.cuda.device_count() > 1:
        #     self.dqn_target = torch.nn.DataParallel(self.dqn_target)
        
        # if torch.cuda.device_count() > 1:
        #     self.dqn = torch.nn.DataParallel(self.dqn)
            
        # if torch.cuda.device_count() > 1:
        #     self.dqn_target = torch.nn.DataParallel(self.dqn_target)
        
        #count = 0
        # for p in self.dqn.parameters(recurse=True):
        #     print (p.size())
        #     if len(p.size()) ==2:
        #         count += p.size(0)*p.size(1)
        #     if len(p.size()) ==4:
        #         count += p.size(0)*p.size(1)*p.size(2)*p.size(3)
        #     else:
        #         count += len(p)
        # print("Total number of parameters: ", count)
        #torchinfo.summary(self.dqn, (1,3,72,72))
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        for param in self.dqn_target.parameters():
            param.requires_grad = False

        # create optimizer
        self.dqn_optim = optim.Adam(
            self.dqn.parameters(),
            lr=self.optim_cfg.lr_dqn,
            weight_decay=self.optim_cfg.weight_decay,
            eps=self.optim_cfg.adam_eps,
        )

        # init network from file
        #self._init_from_file()

    def _init_from_file(self):
        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)
            
    def choose_random_action(self, weight=None):
        return self.env.action_space.sample()
    
    def reset_frame_array(self):
        self.frame_array = []

    def select_action(self, state, is_personalization=False):
        """Select an action from the input space."""
        action_advised = False
        state = preprocess_state_for_inference(state, device)
        if (self.epsilon > np.random.random() or self.total_step < self.hyper_params.init_random_actions):
            selected_action = self.choose_random_action()
        else:
            self.dqn.eval()
            with torch.no_grad():
                policy_dqn_output = self.dqn(state)
                policy_dqn_output = policy_dqn_output.squeeze(0)
                policy_dqn_output = policy_dqn_output.detach().cpu().numpy()
            self.dqn.train()
            selected_action = np.argmax(policy_dqn_output)
        if is_personalization:
            #Get the transitions till now
            # previous_states, actions, rewards, next_states = self.lstm_buffer.get_trajectory()
            # if previous_states:
            #     previous_states_personalization = deepcopy(np.expand_dims(np.array(previous_states), axis = 1))
            #     previous_states_personalization = np.concatenate((previous_states_personalization , state.unsqueeze(0).unsqueeze(0).detach().cpu().numpy()), axis = 0)
            # else:
            #     previous_states_personalization = state.unsqueeze(0).unsqueeze(0).detach().cpu().numpy()
            #state_input = torch.tensor(np.array(previous_states_personalization)).to(device)
            safety_q_value = []
            #state_input = state_input.unsqueeze(0)
            state_input = state.unsqueeze(0).unsqueeze(0).type(torch.float32).to(device)
            size = state_input.size(0)
            #state_input = state_input.type(torch.float32)
            for act in range(self.args.num_action):
                # if previous_states:
                #     previous_action_personalization = np.expand_dims(np.array(actions), axis = 0)
                #     previous_action_personalization = np.concatenate((previous_action_personalization, np.expand_dims(np.array([act]), axis = 0)), axis = -1)
                #     action_vec = previous_action_personalization
                # else:
                #     action_vec = np.expand_dims(np.array([act]), axis = 0)
                action_vec = np.expand_dims(np.array([act]), axis = 0)
                action_tensor = torch.tensor(action_vec).to(device)
                size = state_input.size(0)
                state_encoding = state_input
                selected_action_q_value, _, attn = self.model(state_encoding, action_tensor, torch.tensor([[size]]))
                selected_action_q_value = selected_action_q_value.detach().cpu().numpy()
                safety_q_value.append(selected_action_q_value[0,-1,0])

            safety_q_value = np.array(safety_q_value)
            if self.episode_step == 1:
                self.previous_q_value = 0
            safety_q_value_diff = safety_q_value - self.previous_q_value
            #adjusted_redistributed_reward = safety_q_value_diff - np.sort(safety_q_value_diff)[1]
            adjusted_redistributed_reward = safety_q_value_diff - min(safety_q_value_diff)
            #Discount factor is set at 0.9
            episoding_discounted_return = calculate_discounted_reward(self.adjusted_episode_lstm_reward, 0.9)
            lstm_temperature = max(0.3, self.get_lstm_temperature(episoding_discounted_return))
            #lstm_temperature = 10
            #lstm_temperature = max(0.1, self.get_lstm_temperature(episoding_discounted_return))
            #lstm_temperature = 0.1
            lstm_std = np.std(safety_q_value_diff)
            lstm_prob = softmax_temperature(safety_q_value_diff, lstm_temperature)
            #dqn_temperature = max(0.02, (1-lstm_temperature))
            dqn_temperature = 0.6
            #policy_dqn_output[valid_actions[np.argmin(safety_q_value_diff[valid_actions])]] = min(policy_dqn_output)
            dqn_prob = softmax_temperature(policy_dqn_output, dqn_temperature)
            combined_probability = self.get_combined_policy(lstm_prob, dqn_prob)
            dqn_action = selected_action
            #Selecting the action from the fused policy
            #selected_action = np.random.choice(valid_actions, p=combined_probability)
            selected_action = np.argmax(combined_probability)
            #Changing the q-value of the previous state
            self.previous_q_value = safety_q_value[selected_action]
            #Appending the redistributed reward to calculate the discounted reward
            self.adjusted_episode_lstm_reward.append(adjusted_redistributed_reward[selected_action])
            #self.adjusted_episode_lstm_reward += adjusted_redistributed_reward[selected_action]
            sorted_safety_q_value = np.argsort(safety_q_value)
            #For debugging purpose only
            # if 18<state[0]<=20:
            #     value = state[0]
            #     print(None)
            # if 30<=state[0]<32:
            #     value = state[0]
            #     print(None)
            # if 8<=state[0]<=10:
            #     value = state[0]
            #     print(None)
            # if 38<=state[0]<=40:
            #     value = state[0]
            #     print(None)
        return selected_action, action_advised
    
    def get_combined_policy(self, lstm_prob, dqn_prob):
        if self.args.policy_fusion == 'product':
            product = np.multiply(lstm_prob, dqn_prob)
            return product/product.sum()
        elif self.args.policy_fusion == 'entropy_weighted':
            lstm_entropy  = entropy(lstm_prob)
            dqn_entropy  = entropy(dqn_prob)
            entropy_sum = lstm_entropy + dqn_entropy
            min_entropy = min(lstm_entropy/entropy_sum, dqn_entropy/entropy_sum)
            return min_entropy*dqn_prob + (1-min_entropy)*lstm_prob
        elif self.args.policy_fusion == 'average':
            return (lstm_prob + dqn_prob)/2
        elif self.args.policy_fusion == 'entropy_threshold':
            lstm_entropy  = entropy(lstm_prob)
            dqn_entropy  = entropy(dqn_prob)
            if lstm_entropy < dqn_entropy:
                min_entropy = lstm_entropy
                min_prob = lstm_prob
                max_prob = dqn_prob
            else:
                min_entropy = dqn_entropy
                min_prob = dqn_prob
                max_prob = lstm_prob
            if min_entropy < dqn_entropy + 0.3:
                return min_prob
            else:
                max_prob     
        else:
            raise NotImplementedError ("Choose either product/entropy_weighted policy fusion method")
    
    def get_lstm_temperature(self, adjusted_episode_lstm_reward):    
        return 7/(1 + np.exp(-10*(adjusted_episode_lstm_reward-0)))
    
    def step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, reward, terminated, truncated, info

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)

    def _get_dqn_loss(self, experiences, gamma):
        """Return element-wise dqn loss and Q-values."""
        return calculate_dqn_loss(
            model=self.dqn,
            target_model=self.dqn_target,
            experiences=experiences,
            gamma=gamma,
            use_double_q_update=self.hyper_params.use_double_q_update,
            reward_clip=self.hyper_params.reward_clip,
            reward_scale=self.hyper_params.reward_scale,
        )

    def update_model(self):
        """Train the model after each episode."""
        # 1 step loss
        if self.use_prioritized:
            experiences_one_step = self.memory.sample(self.per_beta)
            weights, indices = experiences_one_step[-3:-1]
            indices = np.array(indices)
            # re-normalize the weights such that they sum up to the value of batch_size
            weights = weights/torch.sum(weights)*float(self.hyper_params.batch_size)
        else:
            indices = np.random.choice(len(self.memory), size=self.hyper_params.batch_size, replace=False)
            weights = torch.from_numpy(np.ones(shape=(indices.shape[0], 1), dtype=np.float64)).type(torch.FloatTensor).to(device)
            experiences_one_step = self.memory.sample(indices=indices)

        dq_loss_element_wise, q_values = self._get_dqn_loss(experiences_one_step, self.hyper_params.gamma)
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self._get_dqn_loss(experiences_n, gamma)

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            # mix of 1-step and n-step returns
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.w_n_step
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # total loss
        loss = dq_loss

        # q_value regularization (not used when w_q_reg is set to 0)
        if self.optim_cfg.w_q_reg > 0:
            q_regular = torch.norm(q_values, 2).mean() * self.optim_cfg.w_q_reg
            loss = loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward()
        if self.hyper_params.gradient_clip is not None:
            clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        #common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)
        if self.total_step % self.hyper_params.target_network_update_interval == 0:
            common_utils.hard_update(self.dqn, self.dqn_target)
        
        # update priorities in PER
        if self.use_prioritized:
            loss_for_prior = dq_loss_element_wise.detach().cpu().numpy().squeeze()
            new_priorities = loss_for_prior + self.hyper_params.per_eps
            if (new_priorities <= 0).any().item():
                print('[ERROR] new priorities less than 0. Loss info: ', str(loss_for_prior))

            # noinspection PyUnresolvedReferences
            self.memory.update_priorities(indices, new_priorities)

            # increase beta
            fraction = min(float(self.i_episode) / self.args.iteration_num, 1.0)
            self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        return loss.item(), q_values.mean().item()
        
    # def pretrain(self, datapath):
    #     good_list, bad_list = load_list(datapath)
    #     min_length = min(len(bad_list), len(good_list))
    #     good_list, bad_list = bad_list[0:min_length], good_list[0:min_length]
    #     print("Data loaded succesfully")
    #     print("Pretraining starting")
    #     for good, bad in zip(good_list, bad_list):
    #         self.lstm_buffer.add_to_buffer_from_list(good, 1) 
    #         #self.lstm_buffer.add_to_buffer_from_list(inter, -0.5) 
    #         self.lstm_buffer.add_to_buffer_from_list(bad, -1) 
    #     self.lstmtraininghandler.train_lstm(0, True)   
            
    #     for good, bad in zip(good_list, bad_list):    
    #        redistributed_reward, _, _ = self.lstmtraininghandler.get_rudder_prediction(np.expand_dims(good[0],0),
    #                                                                             np.expand_dims(good[1],0))
    #        self.safetydqnhandler.add_transition(good[0], good[1], redistributed_reward, good[3], good[4], good[5])
    #        redistributed_reward, _, _ = self.lstmtraininghandler.get_rudder_prediction(np.expand_dims(bad[0],0),
    #                                                                             np.expand_dims(bad[1],0))
    #        self.safetydqnhandler.add_transition(bad[0], bad[1], redistributed_reward, bad[3], bad[4], bad[5])
    #     self.safetydqnhandler.safety_dqn_training(is_pretrain =True)
    #     print("Pretraining done")
    
    def load_params(self, path, device):
        """Load model and optimizer parameters."""
        #path = path + '/dqn_model' + '.tar'
        params = torch.load(path, map_location=device)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the DQN model and optimizer from", path)
    
    def run_episode(self, is_dqn_training = True, is_personalization = False):
        self.previous_state = self.env.reset()[0]
        #self.previous_state = np.array(self.previous_state._frames).flatten()
        #previous_encoding = get_state_one_hot(previous_state[0]['agent'])
        self.episode_step = 0
        losses = list()
        done = False
        score = 0
        reward = 0
        key_state = []
        self.episode_frame_array = []
        self.oracle.reset_episode_count()
        while not done:
            self.total_step += 1
            self.episode_step += 1
            action, _ = self.select_action(self.previous_state, is_personalization)
            self.next_state, reward, terminated, truncated, info = self.step(action)
            #time.sleep(0.05)
            #self.next_state = np.array(self.next_state._frames).flatten()
            done = terminated + truncated
            feedback = self.oracle.get_feedback(self.next_state, self.args.mode)
            original_reward = reward
            self.lstm_buffer.add_transitions(self.previous_state, action, feedback, self.next_state, key_state)
            # Save the new transition
            transition = (self.previous_state, action, reward, self.next_state, done, info)
            self.previous_state = self.next_state
            if is_dqn_training:
                self._add_transition_to_memory(transition)
            if len(self.memory) >= self.hyper_params.update_starts_from and is_dqn_training:
                if self.total_step % self.hyper_params.train_freq == 0:
                    for _ in range(self.hyper_params.multiple_update):
                        loss = self.update_model()
                        losses.append(loss)  # for logging
            score += original_reward
            if self.is_write_video:
                frame = self.env.render(mode='rgb_array')
                frame = to_grayscale(frame)
                self.episode_frame_array.append(frame)
        # print("Score", score, "Step", self.episode_step)
        # time.sleep(1)
        return losses, self.episode_step, score, self.total_step
    
    def run_game(self):
        self.success = 0
        pbar_dqn = tqdm(total=self.args.iteration_num)
        for i_episode in range(1, self.args.iteration_num + 1):
            self.i_episode = i_episode
            start_time = time.time()
            losses, self.episode_step, score, self.total_step = self.run_episode(True)
            
            #Adding the trajectory to the lstm_buffer
            self.lstm_buffer.add_to_buffer()
            #Reset the transition list
            self.lstm_buffer.reset_arrays()
            self.do_post_episode_update()
            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, 
                             avg_loss[0], avg_loss[1], 
                             self.epsilon, score, self.episode_step)
                write_log(self.logger, self.log_dict, log_value, self.args.use_logger)
            if i_episode % self.args.save_freequency == 0:
                params = {
                            "dqn_state_dict": self.dqn.state_dict(),
                            "dqn_target_state_dict": self.dqn_target.state_dict(),
                            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
                        }
                save_models(params, self.dump_dir, 'dqn_model_{}'.format(i_episode))
            pbar_dqn.update(1)
            pbar_dqn.set_description("Score {}".format(score))
        pbar_dqn.close()    
                    
    def train(self):
        print("[INFO] Dump dir: ", self.dump_dir)
        self.oracle = HumanOracle(self.env, self.args.mode)
        self.plot_internals = PlotInternal()
        
        self.lstm_buffer = LSTMBufferHandler(self.lstm_config, self.args.max_steps, self.args.iteration_num)
            
        if self.args.dqn_training:
            self.run_game()
            #Dump the trajectories and model weights
            params = {
            "dqn_state_dict": self.dqn.state_dict(),
            "dqn_target_state_dict": self.dqn_target.state_dict(),
            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
            }
            save_models(params, self.dump_dir, 'dqn_model')
            self.lstm_buffer.dump_buffer_data(self.args.dump_dir, self.args.mode)
        else:
            self.load_params(self.args.model_path, device)
            self.lstm_buffer.fill_buffer_from_disk(self.args.load_data_dir, self.args.mode)
            
    
        if not self.args.is_personalization:
            #Closing the env
            self.env.close
            if self.args.use_logger:
                self.logger.finish_run()
        else:
            if self.args.pref_model == 'LSTM':
                self.traininghandler = LSTMTrainingHandler(self.lstm_config, self.lstm_buffer, self.dump_dir, self.logger)
                print("[INFO] LSTM Preference training initiated")
            else:
                self.traininghandler = TrainingHandler(self.lstm_config, self.lstm_buffer, self.dump_dir, self.logger)
                print("[INFO] PEBBLE Preference training initiated")
            #Training the rudder with the data collected from human and DQN policy learning 
            self.traininghandler.train()
            #Create directory to save videos
            self.video_save_dir = create_dump_directory(self.dump_dir + '/Advise_videos')
            #self.env.reset_arrays()
            self.lstm_buffer.reset_list(self.args.personalization_num_episode)
            self.model = self.traininghandler.get_model()
            #Dumping the Reward model
            if self.args.pref_model == "LSTM":
                checkpoint = {"lstm_weight" : self.model.state_dict()}
                path = self.dump_dir + '/lstm.tar'
            else:
                checkpoint = {"pebble_weight" : self.model.state_dict()}
                path = self.dump_dir + '/pebble.tar'
            torch.save(checkpoint, path)
            #We don't want to do exploration after DQN has trained
            self.epsilon = 0
            self.total_step = self.hyper_params.init_random_actions + 1 #This condition in enforced to get DQN actions
            #Create a dataframe to log wrong action selection
            #self.df = pd.DataFrame(columns=["state_cordinate", "dqn_action", "selected_action", "valid_actions", "lstm_prob", "dqn_prob", "combined_probability", "lstm_temperature", "dqn_temperature"])
            #Wrap the environment to save the test videos
            #self.env = record_videos(self.env, video_folder=self.dump_dir)
            self.model.eval()
            #Starting a new environment with infinite horizon
            #self.env.change_truncate()
            self.env.change_max_step()
            
            self.is_write_video = True
            p_cummulative_prefered_region = 0
            p_cummulative_avoid_region = 0
            d_cummulative_prefered_region = 0
            d_cummulative_avoid_region =0
            save_path = self.dump_dir + '/DQN_Policy'
            os.makedirs(save_path)
            for episode in range(self.args.personalization_num_episode):
                print("Episode: ", episode)
                self.adjusted_episode_lstm_reward = []
                #self.adjusted_episode_lstm_reward = 0
                self.episode = episode
                # print("Running personalised policy")
                _, _, p_score, _ = self.run_episode(False, True)
                write_video(self.episode_frame_array, episode, self.video_save_dir)
                self.lstm_buffer.reset_arrays()
                #Get episode violations count (For logging)
                if self.args.mode == 'both':
                    _, _, p_episode_prefered_region, p_episode_avoid_region = self.oracle.return_counts()
                    p_cummulative_prefered_region += p_episode_prefered_region
                    p_cummulative_avoid_region += p_episode_avoid_region
                elif self.args.mode == 'preference':
                    _, p_episode_prefered_region = self.oracle.return_counts()
                    p_cummulative_prefered_region += p_episode_prefered_region
                elif self.args.mode == 'avoid':
                    _, p_episode_avoid_region = self.oracle.return_counts()
                    p_cummulative_avoid_region += p_episode_avoid_region
                # print("Episode steps p: ", self.episode_step)
                # print("Score p: ", p_score)
                # print("Running DQN policy")
                _, _, d_score, _ = self.run_episode(False, False)
                write_video(self.episode_frame_array, episode, save_path)
                # print("Episode steps DQN: ", self.episode_step)
                # print("Score DQN: ", d_score)
                if self.args.mode == 'both':
                    _, _, d_episode_prefered_region, d_episode_avoid_region = self.oracle.return_counts()
                    d_cummulative_prefered_region += d_episode_prefered_region
                    d_cummulative_avoid_region += d_episode_avoid_region
                elif self.args.mode == 'preference':
                    _, d_episode_prefered_region = self.oracle.return_counts()
                    d_cummulative_prefered_region += d_episode_prefered_region
                elif self.args.mode == 'avoid':
                    _, d_episode_avoid_region = self.oracle.return_counts()
                    d_cummulative_avoid_region += d_episode_avoid_region
                if self.args.mode == 'preference':
                    new_log_value = (episode, p_score, p_cummulative_prefered_region, p_episode_prefered_region, d_score, d_cummulative_prefered_region, d_episode_prefered_region)
                elif self.args.mode == 'avoid':
                    new_log_value = (episode, p_score, p_cummulative_avoid_region, p_episode_avoid_region, d_score, d_cummulative_avoid_region, d_episode_avoid_region)
                elif self.args.mode == 'both':
                    new_log_value = (episode, p_score, p_cummulative_prefered_region, p_cummulative_avoid_region, p_episode_prefered_region, p_episode_avoid_region, d_score, 
                                     d_cummulative_prefered_region, d_cummulative_avoid_region, d_episode_prefered_region, d_episode_avoid_region)
                write_log(self.logger, self.personalized_policy_log_dict, new_log_value, self.args.use_logger)
                
            #Dumping log data as csv file
            #self.df.to_csv(self.dump_dir + '/Visitation.csv')  
            #Running DQN policy alone to see improvement. Epsilon is already set to 0
            self.env.close()
            if self.args.use_logger:
                self.logger.finish_run()

    def do_post_episode_update(self, *argv):
        if self.total_step >= self.hyper_params.init_random_actions:
            # decrease epsilon
            self.epsilon = max(self.min_epsilon, self.hyper_params.epsilon_decay * self.epsilon)
            
    def do_safety_advise_threshold_update(self):
        if self.total_step >= self.hyper_params.init_random_actions:
            self.safety_advising_threshold = max(self.min_safety_advising_threshold, 
                                                 self.safety_advising_threshold_decay * self.safety_advising_threshold)
            


