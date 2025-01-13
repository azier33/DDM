import torch.utils.data as data
from PIL import Image
import os
import h5py
import torch

class SevirDataset(data.Dataset):
    # 新数据集的属性
    urls = ['/raid/DataSet/SevirLr/data/vil']
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'sevir_train.pt'
    test_file = 'sevir_test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.download = download
        self.file_list = self.get_file_list(is_train = train)
        self.vil_array, self.vil_diff = self.load_data(self.file_list)

    def get_file_list(self, is_train):
        file_list = []
        year_path = None
        if is_train:
            year_path = os.path.join(self.root, 'train')    
        else:
            year_path = os.path.join(self.root, 'test')
        if os.path.exists(year_path):
            for filename in os.listdir(year_path):
                if filename.endswith('.h5'):
                    file_list.append(os.path.join(year_path, filename))
        return file_list

    def load_data(self, file_list):
        self.vil_diff = []
        self.vil_array = []
        for file in file_list:
            vil_array_diff = []
            with h5py.File(file, 'r') as hdf5_file:
                vil_data = hdf5_file["vil"] 
                # 读取数据
                self.array = vil_data[:]
                self.array = torch.from_numpy(self.array)/255.0
                for ix in range(1, self.array.shape[3]):
                    diff_data = self.array[:, : , :, ix] - self.array[:, :, :, ix-1]
                    vil_array_diff.append(diff_data)
            self.diff = torch.stack(vil_array_diff, dim=3)
            self.vil_diff.append(self.diff)
            self.vil_array.append(self.array)
        self.vil_array = torch.cat(self.vil_array, dim=0)
        self.vil_diff = torch.cat(self.vil_diff, dim=0)
        return self.vil_array, self.vil_diff

    def __getitem__(self, index):

        data, diff_data = self.vil_array[index,:,:,:], self.vil_diff[index,:,:,:]
        return data, diff_data

    def __len__(self):
        return len(self.vil_array)

    def _check_exists(self):
        # 检查数据集文件是否存在
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def __repr__(self):
        # 描述数据集
        return f"Dataset: SEVIR ({'train' if self.train else 'test'}), Root: {self.root}"
