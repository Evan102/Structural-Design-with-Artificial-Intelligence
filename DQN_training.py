import sys
sys.path.append('Utils/')
sys.path.append('Searching/')
sys.path.append('Models/')


from Searching import MonteCarloTreeSearch
from Searching import Agent_2
from Searching import Environment

import json

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DQN', type=str, required = True)
    parser.add_argument('--batch_size', '-bs', type=int, default = 64)
    parser.add_argument('--path_model', '-pm',
                        help="path to model to continue training", type=str, default = None)
    parser.add_argument('-pms', '--path_model_save',
                        help="path to trained model save directory", type=str, required=True)
    parser.add_argument('--buffer_size', help="buffer size", type = int, default = 1000)
    parser.add_argument('--ground_motion_number', type = int, default = 3)
    
    parser.add_argument('--failed_score', type = float, default = -0.5)
    parser.add_argument('--number_episodes', help = "# of training loop", type = int, default = 10000)
    
    parser.add_argument('--state_plus_index', help = "whether put current index into state", type = bool, default = False)
    parser.add_argument('--decay_factor', type = float, default = 0.99)
    parser.add_argument('--min_epsilon', type = float, default = 1e-2)
    parser.add_argument('--double', type = bool, default = True)
    parser.add_argument('--clipping', type = float, default = 1)
    args = parser.parse_args()
    return args

args = parse_args()

sys.path.append(args.DQN + "/")
# from args.DQN import DQAgent
import DQAgent

history = {}
history["training"] = []

def power_decay_schedule(episode_number: int,
                         decay_factor: float,
                         minimum_epsilon: float) -> float:
    """Power decay schedule found in other practical applications."""
    return max(decay_factor**episode_number, minimum_epsilon)

_epsilon_decay_schedule_kwargs = {
    "decay_factor": args.decay_factor,
    "minimum_epsilon": args.min_epsilon,
}
epsilon_decay_schedule = lambda n: power_decay_schedule(n, **_epsilon_decay_schedule_kwargs)

import torch
from torch import optim

_optimizer_kwargs = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0,
    "amsgrad": False,
}

optimizer_fn = lambda parameters: optim.Adam(parameters, **_optimizer_kwargs)

simulator_path = "Simulator/2022_05_24__11_56_43"
ground_motion_number = args.ground_motion_number
# Environment
env_args = {"simulator_path": simulator_path, "ground_motion_number": ground_motion_number, "method": "MCTS"}
ENV = Environment.StructureSimulator(**env_args)

ENV.failed_score = args.failed_score

# Agent
agent_args = {"mode": "story", "environment": ENV}
AGENT2 = Agent_2.StructureDesigner(**agent_args)

AGENT2.state_plus_index = args.state_plus_index

import os
### redefine our agent, related to environment
_agent_kwargs = {
    "state_size": AGENT2.get_state_size(),
    "action_size": len(AGENT2.all_sections),
    "number_hidden_units": 64,
    "optimizer_fn": optimizer_fn,
    "epsilon_decay_schedule": epsilon_decay_schedule,
    "batch_size": args.batch_size,
    "buffer_size": args.buffer_size,
    "alpha": 1e-3,
    "gamma": 0.99,
    "update_frequency": 4,
    "double_dqn": args.double,  # True uses Double DQN; False uses DQN 
    "seed": None,
}

double_dqn_agent = DQAgent.DeepQAgent(**_agent_kwargs)

folder = args.path_model_save
if not os.path.exists(folder):
    os.makedirs(folder)

PATH = args.path_model

if PATH:
    double_dqn_agent._online_q_network.load_state_dict(torch.load(PATH)['q-network-state'])
    double_dqn_agent._target_q_network.load_state_dict(double_dqn_agent._online_q_network.state_dict())
    double_dqn_agent._optimizer.load_state_dict(torch.load(PATH)['optimizer-state'])
    double_dqn_agent._alpha = torch.load(PATH)['agent-hyperparameters']["alpha"]
    double_dqn_agent._memory = torch.load(PATH)['agent-hyperparameters']["buffer"]
    # double_dqn_agent._memory.batch_size = torch.load(PATH)['agent-hyperparameters']["batch_size"]
    # double_dqn_agent._memory.buffer_size = torch.load(PATH)['agent-hyperparameters']["buffer_size"]
    double_dqn_agent._gamma = torch.load(PATH)['agent-hyperparameters']["gamma"]
    double_dqn_agent._update_frequency = torch.load(PATH)['agent-hyperparameters']["update_frequency"]
    double_dqn_agent._number_episodes = torch.load(PATH)['agent-hyperparameters']["epsiode"]
    
    # args.number_episodes += double_dqn_agent._number_episodes

with open(os.path.join(folder, "history.json"), "w") as outfile:
    json.dump(history, outfile)

double_dqn_agent._clipping = args.clipping

if args.double:

    double_dqn_scores = DQAgent.train(double_dqn_agent, 
                            AGENT2, ENV, folder,
                            "double-dqn-checkpoint.pth",
                            number_episodes=args.number_episodes,
                            target_score=0.5)

elif not args.double:
    print()
    print("falsefalsefalsefalsefalsefalsefalsefalsefalsefalse")
    print()
    dqn_scores = DQAgent.train(double_dqn_agent, 
                            AGENT2, ENV, folder,
                            "dqn-checkpoint.pth",
                            number_episodes=args.number_episodes,
                            target_score=0.5)

# python training.py -pms withindex --state_plus_index True -bs 128 --buffer_size 2000
# python training.py -pms without_index --state_plus_index False -bs 128 --buffer_size 2000