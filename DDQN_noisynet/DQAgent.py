import typing

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

import Q_algorithm
import Buffer
import os
# import mlflow
import time
import json
import math

class Agent:
    
    def choose_action(self, state: np.array) -> int:
        """Rule for choosing an action given the current state of the environment."""
        raise NotImplementedError
        
    def learn(self, experiences: typing.List[Buffer.Experience]) -> None:
        """Update the agent's state based on a collection of recent experiences."""
        raise NotImplementedError

    def save(self, filepath) -> None:
        """Save any important agent state to a file."""
        raise NotImplementedError
        
    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        """Update agent's state after observing the effect of its action on the environment."""
        raise NotImplementedError

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class QNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(QNetwork, self).__init__()

        self.feature = nn.Linear(in_dim, 128)
        self.noisy_layer1 = NoisyLinear(128, 128)
        self.noisy_layer2 = NoisyLinear(128, 128)
        self.noisy_layer3 = NoisyLinear(128, 128)
        self.noisy_layer4 = NoisyLinear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = F.relu(self.feature(x))
        hidden1 = F.relu(self.noisy_layer1(feature))
        hidden2 = F.relu(self.noisy_layer2(hidden1))
        hidden3 = F.relu(self.noisy_layer3(hidden2))
        out = self.noisy_layer4(hidden3)
        
        return out
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()
        self.noisy_layer3.reset_noise()
        self.noisy_layer4.reset_noise()

class DeepQAgent(Agent):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 number_hidden_units: int,
                 optimizer_fn: typing.Callable[[typing.Iterable[nn.Parameter]], optim.Optimizer],
                 batch_size: int,
                 buffer_size: int,
                 epsilon_decay_schedule: typing.Callable[[int], float],
                 alpha: float,
                 gamma: float,
                 update_frequency: int,
                 double_dqn: bool = False,
                 seed: int = None) -> None:
        """
        Initialize a DeepQAgent.
        
        Parameters:
        -----------
        state_size (int): the size of the state space.
        action_size (int): the size of the action space.
        number_hidden_units (int): number of units in the hidden layers.
        optimizer_fn (callable): function that takes Q-network parameters and returns an optimizer.
        batch_size (int): number of experience tuples in each mini-batch.
        buffer_size (int): maximum number of experience tuples stored in the replay buffer.
        epsilon_decay_schdule (callable): function that takes episode number and returns epsilon.
        alpha (float): rate at which the target q-network parameters are updated.
        gamma (float): Controls how much that agent discounts future rewards (0 < gamma <= 1).
        update_frequency (int): frequency (measured in time steps) with which q-network parameters are updated.
        double_dqn (bool): whether to use vanilla DQN algorithm or use the Double DQN algorithm.
        seed (int): random seed
        
        """
        self._state_size = state_size
        self._action_size = action_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # set seeds for reproducibility
        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # initialize agent hyperparameters
        _replay_buffer_kwargs = {
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "random_state": self._random_state
        }
        self._memory = Buffer.ExperienceReplayBuffer(**_replay_buffer_kwargs)
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._alpha = alpha
        self._gamma = gamma
        self._double_dqn = double_dqn
        
        # initialize Q-Networks
        self._update_frequency = update_frequency
        self._online_q_network = QNetwork(self._state_size, self._action_size)
        self._target_q_network = QNetwork(self._state_size, self._action_size)
        self._synchronize_q_networks(self._target_q_network, self._online_q_network)        
        self._online_q_network.to(self._device)
        self._target_q_network.to(self._device)
        self._target_q_network.eval()

        # initialize the optimizer
        self._optimizer = optimizer_fn(self._online_q_network.parameters())

        # initialize some counters
        self._number_episodes = 0
        self._number_timesteps = 0

    # we should customize our own Q-network here
    def _initialize_q_network(self, number_hidden_units: int) -> nn.Module:
        """Create a neural network for approximating the action-value function."""
        q_network = nn.Sequential(
            nn.Linear(in_features=self._state_size, out_features=number_hidden_units),
            nn.ReLU(),
            NoisyLinear(number_hidden_units, number_hidden_units),
            nn.ReLU(),
            NoisyLinear(number_hidden_units, number_hidden_units),
            nn.ReLU(),
            NoisyLinear(number_hidden_units, number_hidden_units),
            nn.ReLU(),
            NoisyLinear(number_hidden_units, self._action_size)
        )
        return q_network

    @staticmethod
    def _soft_update_q_network_parameters(q_network_1: nn.Module,
                                          q_network_2: nn.Module,
                                          alpha: float) -> None:
        """In-place, soft-update of q_network_1 parameters with parameters from q_network_2."""
        for p1, p2 in zip(q_network_1.parameters(), q_network_2.parameters()):
            p1.data.copy_(alpha * p2.data + (1 - alpha) * p1.data)

    @staticmethod
    def _synchronize_q_networks(q_network_1: nn.Module, q_network_2: nn.Module) -> None:
        """In place, synchronization of q_network_1 and q_network_2."""
        _ = q_network_1.load_state_dict(q_network_2.state_dict())

    def _uniform_random_policy(self, state: torch.Tensor, element_type: int) -> int:
        """Choose an action uniformly at random."""
        if element_type == 1:
            return self._random_state.randint(9)
        elif element_type == 0:
            return self._random_state.randint(5) + 9
        # return self._random_state.randint(self._action_size)

    def _greedy_policy(self, state: torch.Tensor, element_type: int) -> int:
        """Choose an action that maximizes the action_values given the current state."""
        # indices = torch.tensor([i for i in range(9)]).to(self._device)
        # print(indices)
        # print((self._online_q_network(state)))
        # print(torch.index_select(self._online_q_network(state), 1, indices))
        if element_type == 1: # which means design beam
            indices = torch.tensor([i for i in range(9)]).to(self._device)
            action = (torch.index_select(self._online_q_network(state), 1, indices)
                      .argmax()
                      .cpu()  # action_values might reside on the GPU!
                      .item())
            # print(f"beam: {action}")
        elif element_type == 0: # which means design column
            indices = torch.tensor([i for i in range(9, 14)]).to(self._device)
            action = (torch.index_select(self._online_q_network(state), 1, indices)
                      .argmax()
                      .cpu()  # action_values might reside on the GPU!
                      .item())
            action += 9
            # print(f"col: {action}")

        return action

    def _epsilon_greedy_policy(self, state: torch.Tensor, epsilon: float, element_type: int) -> int:
        """With probability epsilon explore randomly; otherwise exploit knowledge optimally."""
        # if self._random_state.random() < epsilon:
        #     action = self._uniform_random_policy(state, element_type)
        # else:
        #     action = self._greedy_policy(state, element_type)
        action = self._greedy_policy(state, element_type)
        
        return action

    def choose_action(self, state: np.array, element_type: int) -> int:
        """
        Return the action for given state as per current policy.
        
        Parameters:
        -----------
        state (np.array): current state of the environment.
        
        Return:
        --------
        action (int): an integer representing the chosen action.

        """
        # need to reshape state array and convert to tensor
        state_tensor = (torch.from_numpy(state)
                             .unsqueeze(dim=0)
                             .to(self._device))
            
        # choose uniform at random if agent has insufficient experience
        # if not self.has_sufficient_experience():
        #     action = self._uniform_random_policy(state_tensor, element_type)
        # else:
        #     epsilon = self._epsilon_decay_schedule(self._number_episodes)
        epsilon = 0
        action = self._epsilon_greedy_policy(state_tensor, epsilon, element_type)
        
        return action

    def _greedy_policy_for_inference(self, state, element_type) -> int:
        """Choose an action that maximizes the action_values given the current state."""
        state_tensor = (torch.from_numpy(state)
                             .unsqueeze(dim=0)
                             .to(self._device))
        
        if element_type == 1: # which means design beam
            indices = torch.tensor([i for i in range(9)]).to(self._device)
            action = (torch.index_select(self._online_q_network(state_tensor), 1, indices)
                      .argmax()
                      .cpu()  # action_values might reside on the GPU!
                      .item())
            # print(f"beam: {action}")
        elif element_type == 0: # which means design column
            indices = torch.tensor([i for i in range(9, 14)]).to(self._device)
            action = (torch.index_select(self._online_q_network(state_tensor), 1, indices)
                      .argmax()
                      .cpu()  # action_values might reside on the GPU!
                      .item())
            action += 9
        return action

    def learn(self, experiences: typing.List[Buffer.Experience]) -> None:
        """Update the agent's state based on a collection of recent experiences."""
        states, actions, rewards, next_states, dones = (torch.Tensor(vs).to(self._device) for vs in zip(*experiences))
        
        # need to add second dimension to some tensors
        actions = (actions.long().unsqueeze(dim=1))
        
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)
        
        if self._double_dqn:
            target_q_values = Q_algorithm.double_q_learning_update(next_states,
                                                       rewards,
                                                       dones,
                                                       self._gamma,
                                                       self._online_q_network,
                                                       self._target_q_network)
        else:
            target_q_values = Q_algorithm.q_learning_update(next_states,
                                                rewards,
                                                dones,
                                                self._gamma,
                                                self._target_q_network)

        online_q_values = (self._online_q_network(states)
                               .gather(dim=1, index=actions))
        
        # compute the mean squared loss
        loss = F.mse_loss(online_q_values, target_q_values)
        
        # updates the parameters of the online network
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        self._online_q_network.reset_noise()
        self._target_q_network.reset_noise()

        self._soft_update_q_network_parameters(self._target_q_network,
                                               self._online_q_network,
                                               self._alpha)
    
    def has_sufficient_experience(self) -> bool:
        """True if agent has enough experience to train on a batch of samples; False otherwise."""
        return len(self._memory) >= self._memory.batch_size
    
    def save(self, filepath: str) -> None:
        """
        Saves the state of the DeepQAgent.
        
        Parameters:
        -----------
        filepath (str): filepath where the serialized state should be saved.
        
        Notes:
        ------
        The method uses `torch.save` to serialize the state of the q-network, 
        the optimizer, as well as the dictionary of agent hyperparameters.
        
        """
        checkpoint = {
            "q-network-state": self._online_q_network.state_dict(),
            "optimizer-state": self._optimizer.state_dict(),
            "agent-hyperparameters": {
                "alpha": self._alpha,
                "buffer": self._memory,
                "batch_size": self._memory.batch_size,
                "buffer_size": self._memory.buffer_size,
                "gamma": self._gamma,
                "update_frequency": self._update_frequency,
                "epsiode": self._number_episodes
            }
        }

        torch.save(checkpoint, filepath)
        
    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        """
        Updates the agent's state based on feedback received from the environment.
        
        Parameters:
        -----------
        state (np.array): the previous state of the environment.
        action (int): the action taken by the agent in the previous state.
        reward (float): the reward received from the environment.
        next_state (np.array): the resulting state of the environment following the action.
        done (bool): True is the training episode is finised; false otherwise.
        
        """
        experience = Buffer.Experience(state, action, reward, next_state, done)
        self._memory.append(experience)
            
        if done:
            self._number_episodes += 1
        else:
            self._number_timesteps += 1
            
            # every so often the agent should learn from experiences
            if self._number_timesteps % self._update_frequency == 0 and self.has_sufficient_experience():
                experiences = self._memory.sample()
                self.learn(experiences)

import collections
import typing

# import gym


# def _train_for_at_most(agent: Agent, env: gym.Env, max_timesteps: int) -> int:
#### bugwei
def _train_for_at_most(agent: Agent, AGENT2, ENV, max_timesteps: int) -> float:
    """Train agent for a maximum number of timesteps."""
    #### bugwei
    # state = env.reset()
    AGENT2.initialize_state()
    state = AGENT2.get_state_for_network()

    score = 0
    prev_reward = 0
    for t in range(max_timesteps):
        element_type = AGENT2.element_category_list[t]
        action = agent.choose_action(state, element_type)
        #### bugwei
        # next_state, reward, done, _ = env.step(action)
        AGENT2.take_action(AGENT2.all_sections[action])
        next_state = AGENT2.get_state_for_network()
        
        reward = ENV.score(AGENT2, AGENT2.get_state(), True)
        temp = prev_reward
        prev_reward = reward
        reward -= temp

        done = AGENT2.is_final_state()

        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    return score

                
# def _train_until_done(agent: Agent, env: gym.Env) -> float:
#### bugwei
def _train_until_done(agent: Agent, AGENT2, ENV) -> float:
    """
    AGENT2: Tony.version
    ENV: Tony.version
    """
    """Train the agent until the current episode is complete."""
    # state = env.reset()
    #### bugwei
    AGENT2.initialize_state()
    state = AGENT2.get_state_for_network()

    score = 0
    prev_reward = 0

    done = False
    while not done:
        element_type = AGENT2.element_category_list[AGENT2.current_index]
        action = agent.choose_action(state, element_type)
        # environment should have step and return next_state, reward, done: bool
        #### bugwei
        # next_state, reward, done, _ = env.step(action)
        AGENT2.take_action(AGENT2.all_sections[action])
        next_state = AGENT2.get_state_for_network()
        
        reward = ENV.score(AGENT2, AGENT2.get_state(), True)
        # if (reward < 0):
        #     print(f"done: {AGENT2.is_final_state()}")
        temp = prev_reward
        prev_reward = reward
        reward -= temp

        done = AGENT2.is_final_state()

        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
    return score

#### bugwei
def train(agent: Agent,
          AGENT2, ENV, folder,
          checkpoint_filepath: str,
          target_score: float,
          number_episodes: int,
          maximum_timesteps = None) -> typing.List[float]:
    """
    Reinforcement learning training loop.
    
    Parameters:
    -----------
    agent (Agent): an agent to train.
    env (gym.Env): an environment in which to train the agent.
    checkpoint_filepath (str): filepath used to save the state of the trained agent.
    number_episodes (int): maximum number of training episodes.
    maximum_timsteps (int): maximum number of timesteps per episode.
    
    Returns:
    --------
    scores (list): collection of episode scores from training.
    
    """

    start_epsiode = agent._number_episodes

    scores = []
    most_recent_scores = collections.deque(maxlen=100)
    start_time = time.time()
    for i in range(start_epsiode, number_episodes):
        # if maximum_timesteps is None:
        #     score = _train_until_done(agent, env)
        # else:
        #     score = _train_for_at_most(agent, env, maximum_timesteps)

        #### bugwei
        if maximum_timesteps is None:
            score = _train_until_done(agent, AGENT2, ENV)
        else:
            score = _train_for_at_most(agent, AGENT2, ENV, maximum_timesteps)
      
        scores.append(score)
        most_recent_scores.append(score)
        
        # mlflow.log_metric("Score", score, i+1)
        average_score = sum(most_recent_scores) / len(most_recent_scores)
        # mlflow.log_metric("Average Score", average_score, i+1)

        history = {}
        history["epsiode"] = i+1
        history["time"] = time.time()-start_time
        history["score"] = score
        # history["reward"] = reward
        history["average_score"] = average_score

        with open(os.path.join(folder, "history.json"), 'r+') as file:
            file_data = json.load(file)
            file_data["training"].append(history)
            # Sets file's current position at offset.
            file.seek(0)
            json.dump(file_data, file)
            # convert back to json.
            # json.dump(file_data, file, indent = 4)

        if score >= target_score:
            print(f"\nEnvironment solved in {i:d} episodes!\tAverage Score: {average_score:.2f}")
            agent.save(os.path.join(folder, "best_" + checkpoint_filepath))
            print("state:")
            print(AGENT2.get_state())
            print("score")
            print(score)
            target_score = score
            # break
        
        # if average_score >= target_score:
        #     print(f"\nEnvironment solved in {i:d} episodes!\tAverage Score: {average_score:.2f}")
        #     agent.save(os.path.join(folder, "best_" + checkpoint_filepath))
        #     print("state:")
        #     print(AGENT2.get_state())
        #     print("score")
        #     print(ENV.score(AGENT2, AGENT2.get_state(), True))
        #     target_score = average_score
            # break
        if (i + 1) % 100 == 0:
            print(f"\rEpisode {i + 1}\tAverage Score: {average_score:.2f}")
            # mlflow.log_metric("Average Score per 100 episodes", average_score, i+1)
            agent.save(os.path.join(folder, str(i)+'_'+checkpoint_filepath))

    return scores

def inference(agent: Agent, AGENT2, ENV):
    # fix the q_network's parameters
    # change to eval mode
    agent._online_q_network.eval()

    AGENT2.initialize_state()
    state = AGENT2.get_state_for_network()
    print(f"original: {AGENT2.get_state()}")

    score = 0
    done = False
    while not done:
        element_type = AGENT2.element_category_list[AGENT2.current_index]
        action = agent._greedy_policy_for_inference(state, element_type)
        # environment should have step and return next_state, reward, done: bool
        #### bugwei
        # next_state, reward, done, _ = env.step(action)
        AGENT2.take_action(AGENT2.all_sections[action])
        next_state = AGENT2.get_state_for_network()
        # reward = ENV.score(AGENT2, AGENT2.get_state(), True)
        done = AGENT2.is_final_state()

        # agent.step(state, action, reward, next_state, done)
        state = next_state
        # score += reward
    
    reward = ENV.score(AGENT2, AGENT2.get_state(), True)
    print(f"score: {reward}")
    print(f"design: {AGENT2.get_state()}")
    print(f"done: {AGENT2.is_final_state()}")