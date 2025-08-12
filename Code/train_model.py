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


def queue_loss1(predictions, targes, S, s, const=0.5):
    predictions = m(predictions)
    cols = torch.arange(predictions.size(1), device='cuda')
    mask1 = (cols.unsqueeze(0) >= s.unsqueeze(1)) & (cols.unsqueeze(0) < S.unsqueeze(1))
    new_out = mask1 * predictions
    # x: (16, 51) tensor
    mask = new_out != 0
    counts = mask.sum(dim=1)

    # Replace zeros with +inf for min, and -inf for max
    x_min = new_out.masked_fill(~mask, float('inf')).min(dim=1).values
    x_max = new_out.masked_fill(~mask, float('-inf')).max(dim=1).values

    diffs = (x_max - x_min).mean()

    # # number of non-zeros per row
    # # replace zeros with NaN for ignoring
    # x_masked = new_out.masked_fill(~mask, float('nan'))
    # # compute mean of non-zero entries
    # means = (new_out * mask).sum(dim=1) / counts
    # # compute population variance manually
    # sq_diff = ((new_out - means.unsqueeze(1)) * mask) ** 2
    # var = sq_diff.sum(dim=1) / counts
    # # std is sqrt of variance
    # stds = var.sqrt()
    # # if count == 1 → set std to 0
    # stds = torch.where(counts > 1, stds, torch.zeros_like(stds)).mean()

    only_S = torch.abs(targes[torch.arange(predictions.shape[0]).to(device), S] - predictions[
        torch.arange(predictions.shape[0]).to(device), S]).sum()

    start_indices = 1 + S
    cols = torch.arange(targes.size(1), device='cuda')
    mask = cols.unsqueeze(0) >= start_indices.unsqueeze(1)  # shape: (3, 4)

    # Apply mask and sum
    sums = (predictions * mask).sum()
    SAE = torch.abs(predictions - targes).sum(axis=1).sum()
    # print(stds.item(),0.1*only_S.item(), const*sums.item(), SAE.item())

    return SAE + 0.1 * only_S + const * sums + diffs * 0.5


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

    return SAE + 2.1 * only_S + const * sums


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
        loss += queue_loss1(model(X_valid), y_valid, X_valid[:, -1].int(), X_valid[:, -2].int())
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

            # output = model(X_valid)

            # S = X_valid[:,-1].int()
            # s = X_valid[:,-2].int()
            # start_indices =  s
            # cols = torch.arange(output.size(1), device='cuda')
            # mask = cols.unsqueeze(0) < start_indices.unsqueeze(1)  # shape: (3, 4)

            # first_s = (output * mask)
            # first_s[first_s == 0] = float('-inf')
            # for ind in range(first_s.shape[0]):
            #     first_s[ind, int(s[ind]): int(s[ind]) + int((S -s)[ind])] = output[ind, int(s[ind])]
            #     first_s[ind,int(S[ind])] =  output[ind, int(s[ind]+1)]

            # predictions = m(first_s)

            predictions = model(X_valid)
            output1 = m(predictions)

            S = X_valid[:, -1].int()
            s = X_valid[:, -2].int()
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

            cols = torch.arange(predictions.size(1), device='cuda')
            mask1 = (cols.unsqueeze(0) >= s.unsqueeze(1)) & (cols.unsqueeze(0) < S.unsqueeze(1))
            new_out = mask1 * predictions
            # x: (16, 51) tensor
            mask = new_out != 0
            only_bet_vals = mask*predictions

            mask = only_bet_vals != 0                          # True where non-zero
            means = (only_bet_vals * mask).sum(dim=1) / mask.sum(dim=1)   # mean per row of non-zeros
            means = means.unsqueeze(1)             # shape (16,1) for broadcasting

            x_new = only_bet_vals.clone()
            x_new[mask] = means.expand_as(only_bet_vals)[mask]
            x_new[~mask] = predictions[~mask]

            # error = (torch.pow(torch.abs(predictions - targes), 1)).sum(axis=1)
            error = (torch.pow(torch.abs(x_new - targes),1)).sum(axis = 1)
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

m = nn.Softmax(dim=1)

def main():


    for ind in range(30):

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
                y = torch.from_numpy(y[:, :])

                return inputs, y

        class Net1(nn.Module):

            def __init__(self, input_size, output_size):
                super().__init__()

                self.fc1 = nn.Linear(input_size, 50)
                self.fc2 = nn.Linear(50, 70)
                self.fc3 = nn.Linear(70, 80)
                self.fc4 = nn.Linear(80, 60)
                self.fc5 = nn.Linear(60, output_size)

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
                x = self.fc5(x)
                return x

        class Net2(nn.Module):

            def __init__(self, input_size, output_size):
                super().__init__()

                self.fc1 = nn.Linear(input_size, 50)
                self.fc2 = nn.Linear(50, 70)
                self.fc3 = nn.Linear(70, 80)
                self.fc4 = nn.Linear(80, output_size)



            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = self.fc4(x)
                return x


        class Net3(nn.Module):

            def __init__(self, input_size, output_size):
                super().__init__()

                self.fc1 = nn.Linear(input_size, 50)
                self.fc2 = nn.Linear(50, 55)
                self.fc3 = nn.Linear(55, 60)
                self.fc4 = nn.Linear(60, 55)
                self.fc5 = nn.Linear(55, 55)
                self.fc6 = nn.Linear(55, output_size)


            def forward(self, x):

                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                x = self.fc6(x)
                return x

        path_dump = r'C:\Users\Eshel\workspace\data\batches_10_moms_inv_no_neg'

        file_list = os.listdir(path_dump)
        data_paths = [os.path.join(path_dump, name) for name in file_list]

        num_arrival_moms, num_ser_moms = np.random.choice([ 3, 4, 5, 6]), np.random.choice([ 3, 4, 5, 6])



        batch_size = np.random.choice([1, 2, 4, 8])

        net_arch = np.random.choice([1, 2, 3])

        print(num_arrival_moms, num_ser_moms, batch_size, net_arch)



        dataset = my_Dataset(data_paths, num_arrival_moms, num_ser_moms)
        batch_size = batch_size
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=int(batch_size),
                                  shuffle=True,
                                  num_workers=0)

        # get first sample and unpack
        first_data = dataset[0]
        features, labels = first_data

        path_valid_batch = r'C:\Users\Eshel\workspace\data\valid_batch_10'
        files = os.listdir(path_valid_batch)
        data_paths_valid = [os.path.join(path_valid_batch, name) for name in files]

        dataset_valid = my_Dataset(data_paths_valid, num_arrival_moms, num_ser_moms)
        batch_size = batch_size
        valid_loader = DataLoader(dataset=dataset_valid,
                                  batch_size=int(batch_size),
                                  shuffle=True,
                                  num_workers=0)

        # get first sample and unpack
        first_data = dataset_valid[1]
        features, labels = first_data

        m = nn.Softmax(dim=1)
        input_size = features.shape[1]
        output_size = labels.shape[1]

        if net_arch == 1:
            net = Net1(input_size, output_size).to(device)
        elif net_arch == 2:
            net = Net2(input_size, output_size).to(device)
        elif net_arch == 3:
            net = Net3(input_size, output_size).to(device)



        weight_decay = 5
        curr_lr = 0.001
        EPOCHS = 200
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
        valid_SAE = []

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
                    output = net(X)
                    loss = queue_loss1(output, y, X[:, -1].int(), X[:, -2].int())  # 1 of two major ways to calculate loss
                    # print('loss', loss)
                    loss.backward()
                    optimizer.step()
                    net.zero_grad()

                    if i % 2000 == 2001:
                        with torch.no_grad():

                            for X_valid, y_valid in valid_loader:
                                break

                            X_valid = X_valid.float()
                            X_valid = X_valid.reshape(-1, X_valid.shape[2])
                            y_valid = y_valid.float()
                            y_valid = y_valid.reshape(-1, y_valid.shape[2])
                            X_valid = X_valid.to(device)
                            y_valid = y_valid.to(device)
                            output = net(X_valid)
                            # S = X_valid[:,-1].int()
                            # s = X_valid[:,-2].int()
                            # start_indices =  s
                            # cols = torch.arange(output.size(1), device='cuda')
                            # mask = cols.unsqueeze(0) < start_indices.unsqueeze(1)  # shape: (3, 4)

                            # first_s = (output * mask)
                            # first_s[first_s == 0] = float('-inf')
                            # for ind in range(first_s.shape[0]):
                            #     first_s[ind, int(s[ind]): int(s[ind]) + int((S -s)[ind])] = output[ind, int(s[ind])]
                            #     first_s[ind,int(S[ind])] =  output[ind, int(s[ind]+1)]

                            # output1 = m(first_s)

                            output = net(X_valid)
                            output1 = m(output)

                            S = X_valid[:, -1].int()
                            s = X_valid[:, -2].int()
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

                            ##################################################
                            predictions = new_output
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
                            ###############################################

                            # output1 = new_output

                            ind = np.random.randint(X_valid.shape[0])
                            ## Plotting the graph
                            SAE = torch.abs(output1[ind] - y_valid[ind]).sum().cpu()
                            Demand_scv = (torch.exp(X_valid[:, 1]) - 1).cpu()
                            Leadtime_SCV = ((torch.exp(X_valid[:, 1 + num_arrival_moms]) - torch.exp(
                                X_valid[:, num_arrival_moms]) ** 2) / torch.exp(X_valid[:, num_arrival_moms]) ** 2).cpu()
                            rhos = torch.exp(X_valid[:, num_arrival_moms]).cpu()

                            fig, (ax1) = plt.subplots(1, 1, figsize=(11, 3.5))
                            width = 0.25
                            max_probs = int(X_valid[ind, -1].item())
                            ratio = torch.exp(X_valid[ind, num_arrival_moms]).cpu().item()
                            rects1 = ax1.bar(1.5 * width + np.arange(min(50, max_probs + 5)),
                                             output1[ind, :min(50, max_probs + 5)].cpu(), width, label='NN')
                            rects2 = ax1.bar(np.arange(min(50, max_probs + 5)), y_valid[ind, :min(50, max_probs + 5)].cpu(),
                                             width, label='Label')
                            plt.rcParams['font.size'] = '20'

                            # # Add some text for labels, title and custom x-axis tick labels, etc.
                            ax1.set_ylabel('PMF', fontsize=21)
                            ax1.set_xlabel('Number of customers', fontsize=20)
                            ax1.set_title('Presenting L distribution', fontsize=21, fontweight="bold")
                            ax1.set_xticks(np.linspace(0, min(50, max_probs + 5), min(50, max_probs + 5) + 1).astype(int))
                            ax1.set_xticklabels(
                                np.linspace(0, min(50, max_probs + 5), min(50, max_probs + 5) + 1).astype(int), fontsize=14)
                            ax1.legend(fontsize=22)
                            plt.title(
                                'ind = {}, s = {}, S =  {}, SAE = {:.4f}, SCV_D = {:.4f}, SCV_L = {:.4f}, rho = {:.4f}'.format(
                                    ind, str(X_valid[ind, -2].item()), str(X_valid[ind, -1].item()), SAE.item(),
                                    Demand_scv[ind].item(), Leadtime_SCV[ind].item(), rhos[ind].item()), fontsize=14)
                            plt.show()

            # print('Compute validation')
            loss_list.append(loss.item())
            valid_list.append(valid(valid_loader, net).item())
            valid_SAE.append(compute_sum_error(valid_loader, net).item())
            # for folder in os.listdir(test_paths):

            #     valid_list[folder].append(valid(valid_loader[folder], net, num_ser_moms).item())
            #     compute_sum_error_list[folder].append(compute_sum_error(valid_loader[folder], net, num_ser_moms, False).item())
            print(
                f"epoch: {epoch} ,loss: {loss_list[-1]}, loss valid: {valid_list[-1]}, valid_SAE: {valid_SAE[-1]}, time: {time.time() - t_0}")

            if len(loss_list) > 8:
                if check_loss_increasing(valid_list):
                    curr_lr = curr_lr * lr_first
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                    print(curr_lr)
                else:
                    curr_lr = curr_lr * lr_second
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=(1 / 10 ** weight_decay))
                    print(curr_lr)
            if sys.platform == 'linux':
                model_path = '/scratch/eliransc/inv/models'
                model_results_path = '/scratch/eliransc/inv/results'
            else:
                model_path = r'C:\Users\Eshel\workspace\inventory_training\inventory_models'
                model_results_path = r'C:\Users\Eshel\workspace\inventory_training\inventory_results'

            file_name = 'invnoneg_netarch_' + str(net_arch) + '_batchsize_'  + str(batch_size)  + '_demand_' +str(num_arrival_moms) + '_leadtime_' +str(num_ser_moms) + '.pkl'

            file_name_model  = 'model_'  + file_name
            file_name_model_result = 'res_'  + file_name
            torch.save(net.state_dict(), os.path.join(model_path, file_name_model))
            pkl.dump((valid_SAE, valid_list, loss_list), open(os.path.join(model_results_path,file_name_model_result), 'wb'))


if __name__ == '__main__':
    main()