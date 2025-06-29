from __future__ import print_function
import sys

#防止其他文件不能import本文件内的内容
sys.path.append('.')
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch



class MovingMNIST(data.Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = ['https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz']
    #urls=[]
    raw_folder = ''
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'


    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set
        self.download = download
        
        urls = ['https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz']
        self.download_process(urls=urls)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +' You can use download=True to download it')

        if self.train:
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))/255.0
            #print(self.train_data.shape)#torch.Size([9000, 20, 64, 64])
            train_size=self.train_data.shape
            train_diff_list=[]
            for i in range(1, train_size[1]):
                diff=self.train_data[:, i, :, :] - self.train_data[:, i-1, :, :]
                train_diff_list.append(diff)
            self.train_diff_data = torch.stack(train_diff_list, dim=1)#torch.Size([9000, 19, 64, 64])
                       
        else:
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))/255.0
            # print(self.test_data.shape)#torch.Size([1000, 20, 64, 64]) max: 1, min: 0
            test_size=self.test_data.shape
            test_diff_list=[]
            for i in range(1, test_size[1]):
                # test_diff=self.test_data[:, i, :, :] - self.test_data[:, i-1, :, :]
                test_diff=self.test_data[:, i, :, :] - self.test_data[:, i-1, :, :]
                test_diff_list.append(test_diff)
            self.test_diff_data = torch.stack(test_diff_list, dim=1)#torch.Size([1000, 19, 64, 64])
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        # need to iterate over time
        def _transform_time(data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = self.transform(img) if new_data is None else torch.cat([self.transform(img), new_data], dim=0)
            return new_data

        if self.train:
            data, diff_data = self.train_data[index,:,:,:], self.train_diff_data[index,:,:,:]
        else:
            data, diff_data = self.test_data[index,:,:,:], self.test_diff_data[index,:,:,:]

        #if self.transform is not None:
        #    seq = _transform_time(seq)
        #if self.target_transform is not None:
        #    target = _transform_time(target)

        return data, diff_data

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download_process(self, urls):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        if self.download:
            for url in urls:
                print('Downloading ' + url)
                data = urllib.request.urlopen(url)
                filename = url.rpartition('/')[2]
                file_path = os.path.join(self.root, self.raw_folder, filename)
                with open(file_path, 'wb') as f:
                    f.write(data.read())
                with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                        gzip.GzipFile(file_path) as zip_f:
                    out_f.write(zip_f.read())
                os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )
        
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str