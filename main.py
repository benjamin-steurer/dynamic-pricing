from data_preprocessing import load_and_preprocess_data
from training_loop import training_loop
from inference_loop import inference_loop
from utils import setup_model_and_environment
import torch

if __name__ == "__main__":
    model, device, optimizer, criterion, env, replay_buffer = setup_model_and_environment()

    df_filtered, X_train, X_test, y_train, y_test = load_and_preprocess_data()
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
    
    epsilon = 0.1  # define epsilon here or load from config
    batch_size = 32  # define batch_size here or load from config
    model_path = 'model.pth'  # define model_path here or load from config
    
    training_loop(model, device, optimizer, criterion, env, replay_buffer, df_filtered, X_train_tensor, y_train_tensor, model_path, epsilon, batch_size)
    inference_loop(model, device, env, replay_buffer, df_filtered, criterion, optimizer, batch_size)