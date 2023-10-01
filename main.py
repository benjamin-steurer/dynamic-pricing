import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import load_and_preprocess_data
from dynamic_pricing_env import DynamicPricingEnv
from dqn_model import DQN
from replay_buffer import ReplayBuffer
import random

env = DynamicPricingEnv()

state = env.reset()
print("Initial state:", state)

# Initialize replay buffer and epsilon value
replay_buffer = ReplayBuffer(1000)
epsilon = 0.1

action = env.action_space.sample()
next_state, reward, done, _ = env.step(action)
print("Next state:", next_state)
print("Reward:", reward)

df_filtered, X_train, X_test, y_train, y_test = load_and_preprocess_data()

input_dim = 2  # This should match the shape of the state
output_dim = 1  # Price
model = DQN(input_dim, output_dim)

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Training loop with environment and experience replay
batch_size = 32
for episode in range(1000):
    state = env.reset()
    episode_reward = 0

    for t in range(200):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        # Take action and get reward
        next_state, reward, done, _ = env.step(action)

        # Store experience in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # Train the model if enough experiences are available
        if len(replay_buffer) >= batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

            # Convert to PyTorch tensors
            state_batch = torch.FloatTensor(state_batch)
            action_batch = torch.LongTensor(action_batch)
            reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(next_state_batch)
            done_batch = torch.FloatTensor(done_batch)

            # Compute Q-values and next Q-values
            q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
            next_q_values = model(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + 0.99 * next_q_values * (1 - done_batch)

            # Compute loss and update model
            loss = criterion(q_values, expected_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {episode_reward}")

