import torch

def inference_loop(model, device, env, replay_buffer, df_filtered, criterion, optimizer, batch_size):
    state = None  
    next_state = None  
    done = None  
    while True:
        latest_date = df_filtered['Date'].max()
        latest_price = df_filtered[df_filtered['Date'] == latest_date]['Price'].values[0]
        latest_price_float = float(latest_price)

        current_state_tensor = torch.FloatTensor([[latest_price_float, 0.0]]).to(device)

        with torch.no_grad():
            suggested_price_tensor = model(current_state_tensor)
        suggested_price = suggested_price_tensor.item()
        print(f"Suggested price: {suggested_price}")

        real_world_margin = float(input("Enter the real-world contribution margin for the suggested price: "))

        weighting_factor = 1.5  
        scaled_real_world_margin = real_world_margin * weighting_factor

        replay_buffer.push(state, suggested_price, scaled_real_world_margin, next_state, done)

        if len(replay_buffer) >= batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

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

        user_input = input("Do you want to continue? (y/n): ")
        if user_input.lower() == 'n':
            break
