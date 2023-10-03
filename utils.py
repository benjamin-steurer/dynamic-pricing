import torch
import torch.optim as optim
import torch.nn as nn
from dqn_model import DQN
from dynamic_pricing_env import DynamicPricingEnv
from replay_buffer import ReplayBuffer
import os

def setup_model_and_environment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = DynamicPricingEnv()
    replay_buffer = ReplayBuffer(1000)
    
    model_path = 'model.pth'
    input_dim = 2
    output_dim = 1

    if os.path.exists(model_path):
        model = DQN(input_dim, output_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        model = DQN(input_dim, output_dim)
    model.eval()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='mean')

    return model, device, optimizer, criterion, env, replay_buffer
