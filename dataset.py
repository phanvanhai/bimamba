import torch
import numpy as np
import glob
import scipy.io as sio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import tqdm

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

def compute_xrf55_statistics(root_dir):
    """
    Compute global mean/std from XRF55 TRAIN set only.
    """

    dataset = XRF55Dataset(
        root_dir=root_dir,
        split='train',
        mean=None,
        std=None,
    )

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    print("Computing XRF55 mean/std...")

    for filename, _ in tqdm(dataset.samples):

        file = os.path.join(
            dataset.data_dir,
            filename + ".npy"
        )

        x = np.load(file).astype(np.float32)

        # (270,1000)
        x = x.reshape(-1, x.shape[-1])

        total_sum += x.sum()
        total_sq_sum += np.square(x).sum()
        total_count += x.size

    mean = total_sum / total_count

    var = total_sq_sum / total_count - mean ** 2
    std = np.sqrt(var)

    print(f"XRF55 mean = {mean:.6f}")
    print(f"XRF55 std  = {std:.6f}")

    return mean, std

class XRF55Dataset(Dataset):
    def __init__(self, root_dir, split='train', mean=None, std=None,):
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

        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        x = np.load(self.data_dir + '/' + filename + '.npy').astype(np.float32)

        # (3,90,1000)
        # ->
        # (270,1000)
        x = x.reshape(-1, x.shape[-1])
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)

        x = torch.from_numpy(x).float()

        # -----------------------------
        # Adaptive temporal pooling
        # 1000 -> 250
        # -----------------------------
        x = F.adaptive_max_pool1d(
            x,
            output_size=250
        )

        return x, label

def compute_sshar_mean_std(
    root_dir,
    device='esp',
    signal='amp',
):

    import re

    rooms = [
        'room_02'
    ]

    subjects = [
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

    rx_list = [
        'rx_00',
        'rx_01',
        'rx_02',
    ]

    pattern = re.compile(
        r'act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)'
    )

    total_sum = 0.0
    total_sq = 0.0
    total_count = 0

    for room in rooms:

        for subject in subjects:

            folder = os.path.join(
                root_dir,
                room,
                device,
                rx_list[0],
                subject
            )

            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):

                if not file.startswith(signal):
                    continue

                m = pattern.search(file)

                if m is None:
                    continue

                direction = int(m.group(3))
                rep = int(m.group(4))

                # chỉ TRAIN
                if direction == 0:
                    if rep > 8:
                        continue
                else:
                    if rep > 4:
                        continue

                rx_data = []

                for rx in rx_list:

                    f = os.path.join(
                        root_dir,
                        room,
                        device,
                        rx,
                        subject,
                        file
                    )

                    x = np.load(f).astype(np.float32)

                    rx_data.append(x)

                x = np.stack(rx_data)

                x = x.reshape(-1)

                total_sum += x.sum()

                total_sq += np.square(x).sum()

                total_count += x.size

    mean = total_sum / total_count

    std = np.sqrt(
        total_sq / total_count - mean ** 2
    )

    print(f"SSHAR mean = {mean:.6f}")
    print(f"SSHAR std  = {std:.6f}")

    return mean, std

class SSHARDataset(Dataset):
    def __init__(
        self,
        root_dir,
        device='esp',
        signal='amp',
        split='train',
        mean=None,
        std=None,
    ):
        self.root_dir = root_dir
        self.device = device
        self.signal = signal
        self.split = split

        self.mean = mean
        self.std = std

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
                f"{self.signal}_"
                f"act{act:02d}_"
                f"pos{pos:02d}_"
                f"dir{direction:02d}_"
                f"rep{rep:02d}.npy"
            )

            file = os.path.join(
                self.root_dir,
                room,
                self.device,
                rx,
                subject,
                filename,
            )

            x = np.load(file).astype(np.float32)
            rx_data.append(x)

        # --------------------------------------------------
        # Stack RX
        # ESP  : (3,1,56,1000)
        # ASUS : (3,4,56,1000)
        # --------------------------------------------------
        x = np.stack(rx_data)

        # --------------------------------------------------
        # Merge RX & antenna
        # ESP  : (168,1000)
        # ASUS : (672,1000)
        # --------------------------------------------------
        x = x.reshape(-1, x.shape[-1])

        # --------------------------------------------------
        # Global normalization
        # --------------------------------------------------
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        x = torch.from_numpy(x).float()

        # --------------------------------------------------
        # Adaptive temporal pooling
        # 1000 -> 250
        # --------------------------------------------------
        x = F.adaptive_max_pool1d(
            x,
            output_size=250
        )
        y = act - 1
        return x, y