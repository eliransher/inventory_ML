import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import torch
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import matrix_power
from scipy.special import factorial
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import time
import math
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


def queue_loss(predictions, targes, S, const=0.5):
    predictions = m(predictions)

    only_S = torch.abs(targes[torch.arange(predictions.shape[0]).to(device), S] - predictions[
        torch.arange(predictions.shape[0]).to(device), S]).sum()

    start_indices = 1 + S
    cols = torch.arange(targes.size(1), device='cuda')
    mask = cols.unsqueeze(0) >= start_indices.unsqueeze(1)  # shape: (3, 4)

    # Apply mask and sum
    sums = (predictions * mask).sum()
    SAE = torch.abs(predictions - targes).sum(axis=1).sum()
    # print(SAE, only_S, sums)

    return SAE + 0.1 * only_S + const * sums


def valid(dset_val, model):
    loss = 0
    for batch in dset_val:
        X_valid, y_valid = batch
        X_valid = X_valid.float()
        X_valid = X_valid.reshape(-1, X_valid.shape[-1])
        y_valid = y_valid.float()
        y_valid = y_valid.reshape(-1, y_valid.shape[-1])
        X_valid = X_valid.to(device)
        y_valid = y_valid.to(device)

        if torch.sum(torch.isinf(X_valid)).item() == 0:
            model.zero_grad()
            criterion = nn.MSELoss()  # or nn.BCELoss()
            # Example forward pass
            y_pred = torch.sigmoid(model(X_valid))  # keep outputs between 0 and 1
            loss += criterion(y_pred, y_valid)

    return loss / len(dset_val)


def check_loss_increasing(loss_list, n_last_steps=10, failure_rate=0.45):
    try:
        counter = 0
        curr_len = len(loss_list)
        if curr_len < n_last_steps:
            n_last_steps = curr_len

        inds_arr = np.linspace(n_last_steps - 1, 1, n_last_steps - 1).astype(int)
        for ind in inds_arr:
            if loss_list[-ind] > loss_list[-ind - 1]:
                counter += 1

        # print(counter, n_last_steps)
        if counter / n_last_steps > failure_rate:
            return True

        else:
            return False
    except:
        return False


def compute_sum_error(valid_dl, model):
    with torch.no_grad():
        errors = []

        for batch in valid_dl:
            X_valid, y_valid = batch
            X_valid = X_valid.float()
            X_valid = X_valid.reshape(-1, X_valid.shape[-1])
            y_valid = y_valid.float()
            y_valid = y_valid.reshape(-1, y_valid.shape[-1])
            X_valid = X_valid.to(device)
            y_valid = y_valid.to(device)

            targes = y_valid

            predictions = model(X_valid)
            output1 = m(predictions)

            S = X_valid[:, -1].int()
            start_indices = 1 + S
            cols = torch.arange(y_valid.size(1), device='cuda')
            mask = cols.unsqueeze(0) >= start_indices.unsqueeze(1)
            true_mask = cols.unsqueeze(0) < start_indices.unsqueeze(1)
            new_output = output1 * true_mask
            row_indices = torch.arange(true_mask.size(1)).expand_as(true_mask).to(device)
            last_true_indices = (true_mask * row_indices).max(dim=1).values
            rows = torch.arange(new_output.size(0)).view(-1, 1)
            rows = rows.to(device)
            S_ = 1 - new_output.sum(dim=1)
            new_output[rows.squeeze(), last_true_indices.squeeze()] += S_
            predictions = new_output

            error = (torch.pow(torch.abs(predictions - targes), 1)).sum(axis=1)
            errors.append(error.mean())

    return torch.tensor(errors).mean()


def compute_df_valid(valid_dl, model):
    df_list = []

    with torch.no_grad():
        errors = []

        for batch in valid_dl:

            df = pd.DataFrame([])

            X_valid, y_valid = batch
            X_valid = X_valid.float()
            X_valid = X_valid.reshape(-1, X_valid.shape[2])
            y_valid = y_valid.float()
            y_valid = y_valid.reshape(-1, y_valid.shape[2])
            X_valid = X_valid.to(device)
            y_valid = y_valid.to(device)

            targes = y_valid

            predictions = model(X_valid)
            predictions = m(predictions)

            error = (torch.pow(torch.abs(predictions - targes), 1)).sum(axis=1)
            errors.append(error.mean())

            df['SAE'] = error.to('cpu')

            for ind in range(X_valid.shape[0]):

                for mom in range(num_arrival_moms):
                    df.loc[ind, 'arrive_' + str(mom + 1)] = torch.exp(X_valid[ind, mom]).to('cpu').item()

                for mom in range(num_ser_moms):
                    df.loc[ind, 'arrive_' + str(mom + 1)] = torch.exp(X_valid[ind, num_arrival_moms + mom]).to(
                        'cpu').item()

                df.loc[ind, 'num_servers'] = X_valid[ind, -1].to('cpu').item()

                df_list.append(df)

    return pd.concat(df_list)


def compute_mean_abs_val(loader, model):
    res = []
    for i, (X, y) in enumerate(loader):
        with torch.no_grad():
            X = X.float()
            X = X.reshape(-1, X.shape[2])  # X.reshape( X.shape[1], X.shape[2])
            y = y.float()
            y = y.reshape(-1, y.shape[2])  # y.reshape(y.shape[1], y.shape[2])
            X = X.to(device)
            y = y.to(device)

            criterion = nn.MSELoss()  # or nn.BCELoss()
            # Example forward pass
            y_pred = torch.sigmoid(model(X))

            res.append(torch.abs(y - y_pred).mean())

    return torch.tensor(res).mean()


def compute_sum_error1(valid_dl, model):
    with torch.no_grad():
        errors = []

        for batch in valid_dl:
            X_valid, y_valid = batch
            X_valid = X_valid.float()
            X_valid = X_valid.reshape(-1, X_valid.shape[-1])
            y_valid = y_valid.float()
            y_valid = y_valid.reshape(-1, y_valid.shape[-1])
            X_valid = X_valid.to(device)
            y_valid = y_valid.to(device)

            targes = y_valid

            output = model(X_valid)
            S = X_valid[:, -1].int()
            s = X_valid[:, -2].int()

            cols = torch.arange(output.size(1), device='cuda')
            mask = cols.unsqueeze(0) <= S.unsqueeze(1)  # shape: (3, 4)
            logits = output.masked_fill(~mask, float('-inf'))
            output1 = m(logits)

            predictions = output1
            cols = torch.arange(predictions.size(1), device='cuda')
            mask1 = (cols.unsqueeze(0) >= s.unsqueeze(1)) & (cols.unsqueeze(0) < S.unsqueeze(1))
            new_out = mask1 * predictions
            # x: (16, 51) tensor
            mask = new_out != 0
            only_bet_vals = mask * predictions

            mask = only_bet_vals != 0  # True where non-zero
            means = (only_bet_vals * mask).sum(dim=1) / mask.sum(dim=1)  # mean per row of non-zeros
            means = means.unsqueeze(1)  # shape (16,1) for broadcasting

            x_new = only_bet_vals.clone()
            x_new[mask] = means.expand_as(only_bet_vals)[mask]
            x_new[~mask] = predictions[~mask]

            output1 = x_new

            error = (torch.pow(torch.abs(predictions - targes), 1)).sum(axis=1)
            errors.append(error.mean())

    return torch.tensor(errors).mean()

def main():




    import torch
    import torch.nn as nn

    class my_Dataset(Dataset):
        # Characterizes a dataset for PyTorch
        def __init__(self, data_paths, num_arrival_moms=5, num_ser_moms=5):
            self.data_paths = data_paths
            self.num_arrival_moms = num_arrival_moms
            self.num_ser_moms = num_ser_moms

        def __len__(self):
            return len(self.data_paths)

        def __getitem__(self, index):
            x, y = pkl.load(open(self.data_paths[index], 'rb'))

            x1 = x[:, :self.num_arrival_moms]
            x2 = x[:, 10: 10 + self.num_ser_moms]
            x3 = x[:, 20:]
            x = np.concatenate((x1, x2, x3), axis=1)

            inputs = torch.from_numpy(x)
            y = torch.from_numpy(y)

            return inputs, y

    import torch
    import torch.nn as nn
    class Net(nn.Module):

        def __init__(self, input_size, output_size=1):
            super().__init__()

            self.fc1 = nn.Linear(input_size, 50)
            self.fc2 = nn.Linear(50, 70)
            self.fc3 = nn.Linear(70, 70)
            self.fc4 = nn.Linear(70, 60)
            self.fc5 = nn.Linear(60, 10)
            self.fc6 = nn.Linear(10, output_size)

            # self.fc1 = nn.Linear(input_size , 50)
            # self.fc2 = nn.Linear(50, 50)
            # self.fc3 = nn.Linear(50, 55)
            # self.fc4 = nn.Linear(55, 52)
            # self.fc5 = nn.Linear(52, output_size)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = self.fc6(x)
            return x



    for num_moms in range(10,11):

        num_arrival_moms, num_ser_moms = num_moms, num_moms



        batch_size = 8


        print(num_arrival_moms, num_ser_moms, batch_size)
        path_dump = r'C:\Users\Eshel\workspace\data\inv_all_data\batch_fullfilmentrate'
        file_list = os.listdir(path_dump)
        data_paths = [os.path.join(path_dump, name) for name in file_list]

        dataset = my_Dataset(data_paths, num_arrival_moms, num_ser_moms)
        batch_size = 8
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)

        # get first sample and unpack
        first_data = dataset[0]
        features, labels = first_data

        path_valid_batch = r'C:\Users\Eshel\workspace\data\inv_all_data\batch_valid_fullfill'  # r'C:\Users\Eshel\workspace\data\valid_batch_10'
        # path_valid_batch = r'C:\Users\Eshel\workspace\data\inv_all_data\valid_batch_by_S\17' #r'C:\Users\Eshel\workspace\data\valid_batch_10'

        files = os.listdir(path_valid_batch)
        data_paths_valid = [os.path.join(path_valid_batch, name) for name in files]

        dataset_valid = my_Dataset(data_paths_valid, num_arrival_moms, num_ser_moms)
        batch_size = 8
        valid_loader = DataLoader(dataset=dataset_valid,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)

        # get first sample and unpack
        first_data = dataset_valid[1]
        features, labels = first_data
        print(features.shape, labels.shape)

        m = nn.Softmax(dim=1)
        input_size = features.shape[1]
        output_size = labels.shape[1]
        net = Net(input_size, output_size).to(device)
        weight_decay = 5
        curr_lr = 0.0005
        EPOCHS = 74
        now = datetime.now()
        lr_second = 0.99
        lr_first = 0.75
        current_time = now.strftime("%H_%M_%S") + '_' + str(np.random.randint(1, 1000000, 1)[0])
        print('curr time: ', current_time)

        # Construct dataset

        optimizer = optim.Adam(net.parameters(), lr=curr_lr,
                               weight_decay=(1 / 10 ** weight_decay))  # paramters is everything adjustable in model

        loss_list = []
        valid_list = {}

        num_probs_presenet = 20
        loss_list = []
        valid_list = []
        valid_abs_list = []

        for epoch in range(EPOCHS):

            t_0 = time.time()

            for i, (X, y) in enumerate(train_loader):

                X = X.float()
                X = X.reshape(-1, X.shape[2])  # X.reshape( X.shape[1], X.shape[2])
                y = y.float()
                y = y.reshape(-1, y.shape[2])  # y.reshape(y.shape[1], y.shape[2])
                X = X.to(device)
                y = y.to(device)

                if torch.sum(torch.isinf(X)).item() == 0:
                    net.zero_grad()
                    criterion = nn.MSELoss()  # or nn.BCELoss()
                    # Example forward pass
                    y_pred = torch.sigmoid(net(X))  # keep outputs between 0 and 1
                    loss = criterion(y_pred, y)

                    # loss = queue_loss1(output, y, X[:,-1].int(), X[:,-2].int())  # 1 of two major ways to calculate loss
                    # print('loss', loss)
                    loss.backward()
                    optimizer.step()
                    net.zero_grad()

            # valid_abs_list print('Compute validation')
            loss_list.append(loss.item())
            valid_list.append(valid(valid_loader, net).item())
            valid_abs_list.append(compute_mean_abs_val(valid_loader, net))
            # valid_SAE.append(compute_sum_error1(valid_loader, net).item())
            # for folder in os.listdir(test_paths):

            #     valid_list[folder].append(valid(valid_loader[folder], net, num_ser_moms).item())
            #     compute_sum_error_list[folder].append(compute_sum_error(valid_loader[folder], net, num_ser_moms, False).item())
            print(
                f"epoch: {epoch} ,loss: {loss_list[-1]}, loss valid: {valid_list[-1]}, abs valid: {valid_abs_list[-1]}, time: {time.time() - t_0}")

            if len(loss_list) > 8:
                if check_loss_increasing(valid_list):
                    curr_lr = curr_lr * lr_first
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                    print(curr_lr)
                else:
                    curr_lr = curr_lr * lr_second
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                    print(curr_lr)


            model_path =         r'C:\Users\Eshel\workspace\inventory_training\mom_anal\models'
            model_results_path = r'C:\Users\Eshel\workspace\inventory_training\mom_anal\results'


            file_name = 'fullfill_prob_' + '_batchsize_'  + str(batch_size)  + '_demand_' +str(num_arrival_moms) + '_leadtime_' +str(num_ser_moms) + '.pkl'

            file_name_model  = 'model_'  + file_name
            file_name_model_result = 'res_'  + file_name
            torch.save(net.state_dict(), os.path.join(model_path, file_name_model))
            pkl.dump((valid_abs_list, valid_list, loss_list), open(os.path.join(model_results_path,file_name_model_result), 'wb'))


if __name__ == '__main__':
    main()