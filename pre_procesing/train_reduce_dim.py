import os
from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from pre_procesing.ae_for_shrink_dim import AutoencoderShrink


def train(num_epochs, train_loader, model, optimizer, criterion, ae_scheduler, save_weights_path):
    # Train the autoencoder
    for epoch in range(num_epochs):
        epoch_loss = []
        for i, (batch_data, batch_target) in enumerate(train_loader):
            # Forward pass
            batch_data = batch_data.float().detach()
            batch_target = batch_target.float().detach()

            _, outputs = model(batch_data)
            loss = criterion(outputs, batch_target)
            epoch_loss.append(loss)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ae_scheduler.step()
        print(f"AE loss {torch.mean(torch.tensor(epoch_loss))} ")
    save_weights_path = os.path.join(save_weights_path, "weights.pt")
    model.save(save_weights_path)


# def remove_nan(data):
#     if np.any(np.isnan(data)):
#         my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#         data = my_imputer.fit_transform(data)
#
#     return data
#
#
# def load_and_pre_process_data(src_path, target_path):
#     src_data = np.loadtxt(src_path, delimiter=',')
#     target_data = np.loadtxt(target_path, delimiter=',')
#
#     src_data = remove_nan(src_data)
#     target_data = remove_nan(target_data)
#
#     src_data = pre_process_cytof_data(src_data)
#     target_data = pre_process_cytof_data(target_data)
#
#     return src_data, target_data


# Convert the source and target data to PyTorch tensors and pass through the autoencoder
def pre_processing(src_data, target_data
                   , ae_encoding_dim=25
                   , num_epochs=100
                   , batch_size=256,
                   load_weights_path="",
                   save_weights_path=r"C:\Users\avrah\PycharmProjects\UnsuperVisedBer\weights"):
    # Define the input and encoding dimensions
    input_dim = src_data.shape[1]
    src_data = torch.tensor(src_data,dtype=torch.float32)
    target_data = torch.tensor(target_data,dtype=torch.float32)
    model_for_shrinking_data = AutoencoderShrink(input_dim, ae_encoding_dim,hidden_layers=6)

    if load_weights_path != "":
        load_weights_path = os.path.join(load_weights_path, "weights.pt")
        model_for_shrinking_data.from_pretrain(load_weights_path)
    else:
        train_data_ae_tensor = torch.cat((src_data, target_data), dim=0)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(model_for_shrinking_data.parameters(), lr=0.001)
        ae_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        my_dataset = TensorDataset(train_data_ae_tensor, train_data_ae_tensor)
        train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
        # Train the autoencoder
        train(num_epochs, train_loader, model_for_shrinking_data, optimizer, criterion, ae_scheduler, save_weights_path)

    model_for_shrinking_data.eval()
    src_data_dim_reduce = model_for_shrinking_data.encoder(src_data)
    target_data_dim_reduce = model_for_shrinking_data.encoder(target_data)
    src_data_dim_reduce = src_data_dim_reduce.detach().numpy()
    target_data_dim_reduce = target_data_dim_reduce.detach().numpy()

    return src_data_dim_reduce, target_data_dim_reduce,model_for_shrinking_data
