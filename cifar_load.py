from torchvision.datasets import MNIST, EMNIST, STL10, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from PIL import Image
import pandas as pd
from datasets import CIFAR10_truncated, SVHN_truncated, CIFAR100_truncated
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import logging

from torch.utils.data import Dataset

from dataloaders import dataset

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(
        datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(
        datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_STL10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = STL10(root=datadir, split='train',
                       transform=transforms.ToTensor())
    test_data = STL10(root=datadir, split='test',
                      transform=transforms.ToTensor())
    X_train, y_train = train_data.data, train_data.labels
    X_test, y_test = test_data.data, test_data.labels

    return (X_train, y_train, X_test, y_test)


def load_fmnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = FashionMNIST(datadir, True, transform=transforms.ToTensor())
    test_data = FashionMNIST(datadir, False, transform=transforms.ToTensor())
    X_train, y_train = train_data.data, train_data.targets
    X_test, y_test = test_data.data, test_data.targets

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(
        datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(
        datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    # y_train = y_train.numpy()
    # y_test = y_test.numpy()

    return (X_train, y_train, X_test, y_test)


def load_SVHN_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    SVHN_train_ds = SVHN_truncated(
        datadir, split='train', download=True, transform=transform)
    SVHN_test_ds = SVHN_truncated(
        datadir, split='test', download=True, transform=transform)

    X_train, y_train = SVHN_train_ds.data, SVHN_train_ds.target
    X_test, y_test = SVHN_test_ds.data, SVHN_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_skin_data(datadir, train_idxs, test_idxs):  # idxs相对所有data
    CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    all_data_path = 'data/med_classify_dataset/HAM10000_metadata'
    all_data_df = pd.read_csv(all_data_path)
    all_data_df = pd.concat(
        [all_data_df['image_id'], all_data_df['dx']], axis=1)
    train_idxs = torch.load('partition_strategy/skin_train_idxs.pth')
    test_idxs = torch.load('partition_strategy/skin_test_idxs.pth')

    X_train, y_train, X_test, y_test = [], [], [], []
    train_df = all_data_df.iloc[train_idxs]
    test_df = all_data_df.iloc[test_idxs]

    train_names = all_data_df.iloc[train_idxs]['image_id'].values.astype(
        str).tolist()
    train_lab = all_data_df.iloc[train_idxs]['dx'].values.astype(str)
    test_names = all_data_df.iloc[test_idxs]['image_id'].values.astype(
        str).tolist()
    test_lab = all_data_df.iloc[test_idxs]['dx'].values.astype(str)
    for idx in range(len(train_idxs)):
        X_train.append(datadir + 'med_classify_dataset/images/' +
                       train_names[idx] + '.jpg')
        y_train.append(CLASS_NAMES.index(train_lab[idx]))

    for idx in range(len(test_idxs)):
        X_test.append(datadir + 'med_classify_dataset/images/' +
                      test_names[idx] + '.jpg')
        y_test.append(CLASS_NAMES.index(test_lab[idx]))
    return X_train, y_train, X_test, y_test


def record_net_data_stats(y_train, net_dataidx_map):
    # usage: ?
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_parties, labeled_num, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)

    state = np.random.get_state()
    # let X_train and y_train have the same shuffle state
    np.random.shuffle(X_train)
    # print(a)
    # result:[6 4 5 3 7 2 0 1 8 9]
    np.random.set_state(state)
    np.random.shuffle(y_train)
    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        # min_require_size = 100
        sup_size = int(len(y_train) / 10)
        N = y_train.shape[0] - sup_size
        net_dataidx_map = {}
        for sup_i in range(labeled_num):
            net_dataidx_map[sup_i] = [i for i in range(
                sup_i * sup_size, (sup_i + 1) * sup_size)]

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties - labeled_num)]
            for k in range(K):
                idx_k = np.where(
                    y_train[int(labeled_num * len(y_train) / 10):] == k)[0] + sup_size
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(
                    [p * (len(idx_j) < N / (n_parties - labeled_num)) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,
                             idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties - labeled_num):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j + labeled_num] = idx_batch[j]
    return (X_train, y_train, X_test, y_test, net_dataidx_map)
    traindata_cls_counts = record_net_data_stats(
        y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def partition_data_allnoniid(dataset, datadir, train_idxs=None, test_idxs=None, partition="noniid", n_parties=10,
                             beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'SVHN':
        X_train, y_train, X_test, y_test = load_SVHN_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'skin':
        X_train, y_train, X_test, y_test = load_skin_data(
            datadir, train_idxs, test_idxs)
    elif dataset == 'fmnist':
        X_train, y_train, X_test, y_test = load_fmnist_data(datadir)

    if dataset != 'skin':
        n_train = y_train.shape[0]
        if partition == "homo" or partition == "iid":
            idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(idxs, n_parties)
            net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

        elif partition == "noniid-labeldir" or partition == "noniid":
            min_size = 0
            min_require_size = 10
            K = 10

            N = y_train.shape[0]
            net_dataidx_map = {}

            while min_size < min_require_size:
                # repeat this process unitl the min_size of one party is greater than requirement
                idx_batch = [[] for _ in range(n_parties)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(beta, n_parties))
                    # whether the data number in party k is enough
                    proportions = np.array(
                        [p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                    # normalize proportions
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) *
                                   len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j,
                                 idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(n_parties):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

            traindata_cls_counts = record_net_data_stats(
                y_train, net_dataidx_map)
        return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts
    else:
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


# 客户端异构噪声生成器
class ClientBasedNoiseGenerator:
    """
    基于客户端ID生成不同类型和强度的噪声
    每个客户端会有独特但一致的噪声模式
    """

    def __init__(self, num_clients=10, base_noise_level=0.05, max_noise_level=2.0, deterministic_pattern=False):
        """
        初始化噪声生成器

        Args:
            num_clients: 客户端总数
            base_noise_level: 基础噪声水平
            max_noise_level: 最大噪声水平
            deterministic_pattern: 是否使用确定性的噪声模式（用于可复现性）
        """
        self.num_clients = num_clients
        self.base_noise_level = base_noise_level
        self.max_noise_level = max_noise_level
        self.deterministic_pattern = deterministic_pattern

        # 预定义客户端噪声配置
        self.client_configs = self._generate_client_configs()

    def _generate_client_configs(self):
        """为每个客户端生成噪声配置"""
        configs = {}
        noise_types = ['gaussian', 'salt_pepper',
                       'localized', 'spatial_varying']

        for client_id in range(self.num_clients):
            # 基于客户端ID确定噪声类型和强度
            noise_type = noise_types[client_id % len(noise_types)]

            # 噪声强度随客户端ID非线性变化
            intensity_factor = (client_id / max(1, self.num_clients - 1))
            noise_level = self.base_noise_level + intensity_factor * \
                (self.max_noise_level - self.base_noise_level)

            configs[client_id] = {
                'noise_type': noise_type,
                'noise_level': noise_level,
                'seed': client_id * 42 if self.deterministic_pattern else None
            }

        return configs

    def add_gaussian_noise(self, tensor, client_id):
        """添加高斯噪声"""
        config = self.client_configs.get(client_id, self.client_configs[0])
        noise_level = config['noise_level']

        if config['seed'] is not None:
            torch.manual_seed(config['seed'])

        noise = torch.randn_like(tensor) * noise_level
        return tensor + noise

    def add_salt_pepper_noise(self, tensor, client_id):
        """添加椒盐噪声"""
        config = self.client_configs.get(client_id, self.client_configs[0])
        noise_ratio = min(config['noise_level'] * 0.1, 0.2)  # 限制椒盐噪声比例

        if config['seed'] is not None:
            torch.manual_seed(config['seed'])

        mask = torch.rand_like(tensor)
        noisy_tensor = tensor.clone()
        noisy_tensor[mask < noise_ratio / 2] = 0  # 椒噪声
        noisy_tensor[mask > 1 - noise_ratio / 2] = 1  # 盐噪声

        return noisy_tensor

    def add_localized_noise(self, tensor, client_id):
        """添加局部噪声（只在图像某些区域添加噪声）"""
        config = self.client_configs.get(client_id, self.client_configs[0])

        if config['seed'] is not None:
            torch.manual_seed(config['seed'])

        # 根据客户端ID确定噪声区域
        h, w = tensor.shape[-2:]
        region_h = h // 2
        region_w = w // 2

        # 确定噪声区域位置
        start_h = (client_id % 2) * region_h
        start_w = ((client_id // 2) % 2) * region_w

        noisy_tensor = tensor.clone()
        noise = torch.randn(
            tensor.shape[0], region_h, region_w) * config['noise_level']

        if len(tensor.shape) == 3:
            noisy_tensor[:, start_h:start_h + region_h,
                         start_w:start_w + region_w] += noise

        return noisy_tensor

    def add_spatial_varying_noise(self, tensor, client_id):
        """添加空间变化噪声（噪声强度随位置变化）"""
        config = self.client_configs.get(client_id, self.client_configs[0])

        if config['seed'] is not None:
            torch.manual_seed(config['seed'])

        h, w = tensor.shape[-2:]

        # 创建空间变化的噪声强度图
        y_coords = torch.linspace(0, 1, h).unsqueeze(1)
        x_coords = torch.linspace(0, 1, w).unsqueeze(0)

        # 根据客户端ID决定渐变方向
        if client_id % 4 == 0:
            intensity_map = y_coords * x_coords
        elif client_id % 4 == 1:
            intensity_map = (1 - y_coords) * x_coords
        elif client_id % 4 == 2:
            intensity_map = y_coords * (1 - x_coords)
        else:
            intensity_map = (1 - y_coords) * (1 - x_coords)

        noise = torch.randn_like(tensor) * config['noise_level']
        spatial_noise = noise * intensity_map

        return tensor + spatial_noise

    def apply_noise(self, tensor, client_id):
        """根据客户端配置应用相应的噪声"""
        config = self.client_configs.get(client_id, self.client_configs[0])
        noise_type = config['noise_type']

        if noise_type == 'gaussian':
            return self.add_gaussian_noise(tensor, client_id)
        elif noise_type == 'salt_pepper':
            return self.add_salt_pepper_noise(tensor, client_id)
        elif noise_type == 'localized':
            return self.add_localized_noise(tensor, client_id)
        elif noise_type == 'spatial_varying':
            return self.add_spatial_varying_noise(tensor, client_id)
        else:
            return self.add_gaussian_noise(tensor, client_id)


# 创建一个全局的噪声生成器实例，假设有10个客户端
noise_generator = ClientBasedNoiseGenerator(
    num_clients=10, base_noise_level=0.05, max_noise_level=2.0, deterministic_pattern=False)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., client_id=0):
        self.std = std
        self.mean = mean
        self.client_id = client_id

    def __call__(self, tensor):
        return noise_generator.add_gaussian_noise(tensor, self.client_id)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, client_id={2})'.format(self.mean, self.std, self.client_id)


def get_dataloader(args, data_np, label_np, dataset_type, datadir, train_bs, is_labeled=None, data_idxs=None,
                   is_testing=False, pre_sz=40, input_sz=32, noise_level=0, client_id=0):
    if dataset_type == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442],
                                         std=[0.19803012, 0.20101562, 0.19703614])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'
    elif dataset_type == 'skin':
        normalize = transforms.Normalize(mean=[0.7630332, 0.5456457, 0.57004654],
                                         std=[0.14092809, 0.15261231, 0.16997086])
    elif dataset_type == 'cifar10':
        normalize = transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                         std=[0.24703233, 0.24348505, 0.26158768])
        assert pre_sz == 40 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'

    elif dataset_type == 'fmnist':
        normalize = transforms.Normalize(mean=[0.2860402],
                                         std=[0.3530239])
        assert pre_sz == 36 and input_sz == 32, 'Error: Wrong input size for 32*32 dataset'

    if not is_testing:
        if is_labeled:
            trans = transforms.Compose(
                [transforms.RandomCrop(size=(input_sz, input_sz)),
                 transforms.RandomHorizontalFlip(p=0.5),
                 transforms.ToTensor(),
                 AddGaussianNoise(0., noise_level, client_id),
                 normalize
                 ])
            ds = dataset.CheXpertDataset(dataset_type, data_np, label_np, pre_sz, pre_sz, lab_trans=trans,
                                         is_labeled=True, is_testing=False)
        else:

            weak_trans = transforms.Compose([
                transforms.RandomCrop(size=(input_sz, input_sz)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, client_id),
                normalize
            ])
            strong_trans = transforms.Compose([
                transforms.RandomCrop(size=(224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, client_id),
                normalize
            ])

            ds = StrongDataset(dataset_type, data_np, label_np, pre_sz, pre_sz,
                               un_trans_wk=weak_trans,
                               un_trans_st=strong_trans,
                               data_idxs=data_idxs,
                               is_labeled=False,
                               is_testing=False)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs,
                             drop_last=False, shuffle=True, num_workers=16, pin_memory=True)
    else:
        ds = dataset.CheXpertDataset(dataset_type, data_np, label_np, input_sz, input_sz, lab_trans=transforms.Compose([
            # K.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize
        ]), is_labeled=True, is_testing=True)
        dl = data.DataLoader(dataset=ds, batch_size=train_bs, drop_last=False,
                             shuffle=False, num_workers=16, pin_memory=True)
    return dl, ds


class StrongDataset(Dataset):
    def __init__(self, dataset_type, data_np, label_np, pre_w, pre_h, lab_trans=None, un_trans_wk=None, un_trans_st=None,
                 data_idxs=None,
                 is_labeled=False,
                 is_testing=False):
        """
        Args:

            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(StrongDataset, self).__init__()

        self.images = data_np
        self.labels = label_np
        self.is_labeled = is_labeled
        self.dataset_type = dataset_type
        self.is_testing = is_testing

        self.resize = transforms.Compose([transforms.Resize((pre_w, pre_h))])
        self.resize_trans = transforms.Compose([transforms.Resize((256, 256))])
        if not is_testing:
            if is_labeled == True:
                self.transform = lab_trans
            else:
                self.data_idxs = data_idxs
                self.weak_trans = un_trans_wk
                self.strong_trans = un_trans_st
        else:
            self.transform = lab_trans

        print('Total # images:{}, labels:{}'.format(
            len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        if self.dataset_type == 'skin':
            img_path = self.images[index]
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.fromarray(self.images[index]).convert('RGB')

        image_resized = self.resize(image)
        image_resized_trans = self.resize_trans(image)
        label = self.labels[index]

        if not self.is_testing:
            if self.is_labeled == True:
                if self.transform is not None:
                    image = self.transform(image_resized).squeeze()
                    # image=image[:,:224,:224]
                    return index, image, torch.FloatTensor([label])
            else:
                if self.weak_trans and self.data_idxs is not None:
                    weak_aug = self.weak_trans(image_resized)
                    strong_aug = self.weak_trans(image_resized)
                    idx_in_all = self.data_idxs[index]

                    for idx in range(len(weak_aug)):
                        weak_aug[idx] = weak_aug[idx].squeeze()
                        strong_aug[idx] = strong_aug[idx].squeeze()
                    return index, [weak_aug, strong_aug], torch.FloatTensor([label])
        else:
            image = self.transform(image_resized)
            return index, image, torch.FloatTensor([label])
            # return index, weak_aug, strong_aug, torch.FloatTensor([label])

    def __len__(self):
        return len(self.labels)
