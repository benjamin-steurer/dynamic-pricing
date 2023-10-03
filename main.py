import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import load_and_preprocess_data
from dynamic_pricing_env import DynamicPricingEnv
from dqn_model import DQN
from replay_buffer import ReplayBuffer
import random
import numpy as np

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

env = DynamicPricingEnv()

state = env.reset()

replay_buffer = ReplayBuffer(1000)
epsilon = 0.1

action = env.action_space.sample()
next_state, reward, done, _ = env.step(action)
print("Next state:", next_state)
print("Reward:", reward)

df_filtered, X_train, X_test, y_train, y_test = load_and_preprocess_data()

input_dim = 2 
output_dim = 1  # Number of possible actions (prices)
model = DQN(input_dim, output_dim).to(device)  # Move model to device

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Move data to device
X_train_tensor = torch.FloatTensor(X_train.values).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

batch_size = 32
for episode in range(1000):
    state = env.reset()
    episode_reward = 0

    for t in range(200):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(np.array([state])).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(replay_buffer) >= batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

            # Move batch data to device
            state_batch = torch.FloatTensor(state_batch).to(device)
            action_batch = torch.LongTensor(action_batch).to(device)
            reward_batch = torch.FloatTensor(reward_batch).to(device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(device)
            done_batch = torch.FloatTensor(done_batch).to(device)

            q_values = model(state_batch).squeeze(1)
            next_q_values = model(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + 0.99 * next_q_values * (1 - done_batch)

            loss = criterion(q_values, expected_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {episode_reward}")




while True:  # Infinite loop to keep the program running
    latest_date = df_filtered['Date'].max()
    latest_price = df_filtered[df_filtered['Date'] == latest_date]['Price'].values[0]
    latest_price_float = float(latest_price)

    current_state_tensor = torch.FloatTensor([[latest_price_float, 0.0]]).to(device)

    with torch.no_grad():
        suggested_price_tensor = model(current_state_tensor)
    suggested_price = suggested_price_tensor.item()
    print(f"Suggested price: {suggested_price}")

    # New code to accept real-world contribution margin
    real_world_margin = float(input("Enter the real-world contribution margin for the suggested price: "))

    # Use the real-world margin as the reward for the last action
    replay_buffer.push(state, suggested_price, real_world_margin, next_state, done)

    # Update your model based on the new data in the replay buffer
    if len(replay_buffer) >= batch_size:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

        # Move batch data to device
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.LongTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).to(device)

        q_values = model(state_batch).squeeze(1)
        next_q_values = model(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + 0.99 * next_q_values * (1 - done_batch)

        loss = criterion(q_values, expected_q_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Add a way to break the loop if needed
    user_input = input("Do you want to continue? (y/n): ")
    if user_input.lower() == 'n':
        break