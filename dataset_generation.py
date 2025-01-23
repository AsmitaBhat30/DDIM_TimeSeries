import pickle
from torch.utils.data import DataLoader, Dataset
import torch
import pdb

class Forecasting_Dataset(Dataset):
    def __init__(self, data, seq_length=192):
        self.seq_length = seq_length
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_length]

    def __len__(self):
        return len(self.data) - self.seq_length


def load_electricity_data(datatype):
    if datatype == 'electricity':
        datafolder = './data/electricity_nips'
    paths = datafolder + '/data.pkl'

    # shape: (T x N)
    # mask_data is usually filled by 1
    with open(paths, 'rb') as f:
        main_data, _ = pickle.load(f)

    paths = datafolder + '/meanstd.pkl'
    with open(paths, 'rb') as f:
        mean_data, std_data = pickle.load(f)

    main_data = (main_data - mean_data) / std_data
    return main_data


def get_dataloader(datatype, batch_size=8):

    data = load_electricity_data(datatype)
    data = torch.tensor(data, dtype=torch.float32)

    generator1 = torch.Generator().manual_seed(49)
    train_data, valid_data, test_data = torch.utils.data.random_split(data, [0.7, 0.2, 0.1], generator=generator1)
    train_loader = DataLoader(
        Forecasting_Dataset(train_data), batch_size=batch_size, shuffle=1)

    valid_loader = DataLoader(
        Forecasting_Dataset(valid_data), batch_size=batch_size, shuffle=0)

    test_loader = DataLoader(
        Forecasting_Dataset(test_data), batch_size=batch_size, shuffle=0)

    return train_loader, valid_loader, test_loader