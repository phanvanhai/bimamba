import torch
import numpy as np
import glob
import scipy.io as sio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir + '/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir + '/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data), 90, 250)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data

class NTU_HAR_Dataset(Dataset):
    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform
        self.data_list = glob.glob(root_dir + '/*/*.mat')
        self.folder = glob.glob(root_dir + '/*/')
        self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        # normalize
        x = (x - 42.3199) / 4.9802
        x = torch.FloatTensor(x)
        x = x.unsqueeze(0)
        x = F.max_pool1d(x, kernel_size=8, stride=8)
        x = x.squeeze(0)
        if self.transform:
            x = self.transform(x)
        x = torch.FloatTensor(x)
        return x, y

class XRF55Dataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        if split == 'train':
            self.data_dir = root_dir + '/train_data'
            self.label_file = root_dir + '/dml_train.txt'
        else:
            self.data_dir = root_dir + '/test_data'
            self.label_file = root_dir + '/dml_val.txt'
        self.samples = []

        with open(self.label_file, 'r') as f:
            for line in f:
                filename, subject, activity = line.strip().split(',')
                self.samples.append((
                    filename,
                    int(activity) - 31
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        x = np.load(self.data_dir + '/' + filename + '.npy')

        # (3,114,500)
        # ->
        # (342,500)
        x = x.reshape(-1, x.shape[-1])
        x = torch.FloatTensor(x)

        return x, label

class SSHARDataset(Dataset):
    def __init__(
        self,
        root_dir,
        device='esp',
        signal='amp',
        split='train',
    ):
        self.root_dir = root_dir
        self.device = device
        self.signal = signal
        self.split = split
        self.rooms = [
            'room_02'
        ]
        self.subjects = [
            'subject_01',
            'subject_02',
            'subject_03',
            'subject_04',
            'subject_09',
            'subject_10',
            'subject_11',
            'subject_12',
            'subject_13',
            'subject_14',
        ]
        self.rx_list = [
            'rx_00',
            'rx_01',
            'rx_02',
        ]
        self.samples = []
        self.build_index()

    def build_index(self):
        import re
        pattern = re.compile(
            r'act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)'
        )
        for room in self.rooms:
            for subject in self.subjects:
                folder = os.path.join(
                    self.root_dir,
                    room,
                    self.device,
                    self.rx_list[0],
                    subject
                )

                if not os.path.exists(folder):
                    continue

                for file in os.listdir(folder):

                    if not file.startswith(self.signal):
                        continue

                    m = pattern.search(file)

                    if m is None:
                        continue

                    act = int(m.group(1))
                    pos = int(m.group(2))
                    direction = int(m.group(3))
                    rep = int(m.group(4))

                    train = False

                    if direction == 0:
                        train = rep <= 8
                    else:
                        train = rep <= 4

                    if self.split == 'train' and not train:
                        continue

                    if self.split == 'test' and train:
                        continue

                    self.samples.append(
                        (
                            room,
                            subject,
                            act,
                            pos,
                            direction,
                            rep,
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        room, subject, act, pos, direction, rep = self.samples[idx]
        rx_data = []
        for rx in self.rx_list:
            filename = (
                f'{self.signal}_'
                f'act{act:02d}_'
                f'pos{pos:02d}_'
                f'dir{direction:02d}_'
                f'rep{rep:02d}.npy'
            )

            file = os.path.join(
                self.root_dir,
                room,
                self.device,
                rx,
                subject,
                filename,
            )
            x = np.load(file)
            rx_data.append(x)

        x = np.stack(rx_data)

        #
        # ESP
        # (3,1,56,1000)
        #
        # ASUS
        # (3,4,56,1000)
        #
        x = x.reshape(-1, x.shape[-1])

        #
        # ESP
        # (168,1000)
        #
        # ASUS
        # (672,1000)
        #
        x = torch.FloatTensor(x)
        y = act - 1
        return x, y