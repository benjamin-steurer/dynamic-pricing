import torch
import random
import numpy as np
from multiprocessing import Process, Queue

def worker(q, model, device, env, epsilon):
    state = env.reset()
    initial_price = state[0]

    if random.random() < epsilon:
        price_difference = random.choice([-3, -2, -1, 1, 2, 3])
        action = initial_price + price_difference
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        action = q_values.argmax().item()

    next_state, reward, done, _ = env.step(action)

    # Push the experience into the queue
    q.put((state, action, reward, next_state, done))

def training_loop(model, device, optimizer, criterion, env, replay_buffer, df_filtered, X_train_tensor, y_train_tensor, model_path, epsilon=0.2, batch_size=64):
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor).mean()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), model_path)

    for episode in range(1000):
        q = Queue()
        episode_reward = 0 

        processes = [Process(target=worker, args=(q, model, device, env, epsilon)) for _ in range(4)]
        for p in processes:
            p.start()

        for _ in range(4):
            state, action, reward, next_state, done = q.get()
            replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

        for p in processes:
            p.join()

        if len(replay_buffer) >= batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
            state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
            action_batch = torch.LongTensor(action_batch).to(device)
            reward_batch = torch.FloatTensor(reward_batch).to(device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(device)
            done_batch = torch.FloatTensor(done_batch).to(device)

            q_values = model(state_batch).squeeze(1)
            next_q_values = model(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + 0.99 * next_q_values * (1 - done_batch)
            
            loss = criterion(q_values, expected_q_values.detach().unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}")
