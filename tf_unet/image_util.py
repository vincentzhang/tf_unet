# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import glob
import numpy as np
from PIL import Image
import h5py
import os

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 1
    n_class = 2
    

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()
            
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        
        train_data, labels = self._post_process(train_data, labels)
        #import pdb;pdb.set_trace() 
        nx = data.shape[1]
        ny = data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),
    
    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels
        
        return label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        return data, labels
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y
    
class SimpleDataProvider(BaseDataProvider):
    """
    A simple data provider for numpy arrays. 
    Assumes that the data and label are numpy array with the dimensions
    data `[n, X, Y, channels]`, label `[n, X, Y, classes]`. Where
    `n` is the number of images, `X`, `Y` the size of the image.

    :param data: data numpy array. Shape=[n, X, Y, channels]
    :param label: label numpy array. Shape=[n, X, Y, classes]
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, data, label, a_min=None, a_max=None, channels=1, n_class = 2):
        super(SimpleDataProvider, self).__init__(a_min, a_max)
        self.data = data
        self.label = label
        self.file_count = data.shape[0]
        self.n_class = n_class
        self.channels = channels

    def _next_data(self):
        idx = np.random.choice(self.file_count)
        return self.data[idx], self.label[idx]


class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif', shuffle_data=True, n_class = 2):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        
        self.data_files = self._find_data_files(search_path)
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if not self.mask_suffix in name]
    
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.data_files)
        
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)
        
        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)
    
        return img,label

class HDF5DataProvider(BaseDataProvider):
    """
    Data provider for hdf5 images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = HDF5DataProvider("/data/dataset/hip/abhi")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".h5",
                    mask_suffix='_mask.h5', split="train", flip=True,
                    use_empty=False,
                    mean=34.1311, shuffle_data=True, n_class = 2):
        super(HDF5DataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class
        self.data_dir = search_path
        self.split = split
        self.flip = flip
        self.mean = mean

        # handle to the data/label h5 files
        self._image_h5f = h5py.File(os.path.join(self.data_dir,
            'seg_band.h5'), 'r')
        self._label_h5f = h5py.File(os.path.join(self.data_dir,
            'seg_band_mask.h5'), 'r')

        # load indices for images and labels, train or val
        split_f  = '{}/{}.txt'.format(self.data_dir, self.split)
        # h5 indexed by the volume names
        self._vol_names = open(split_f, 'r').read().splitlines()
        self.use_empty = use_empty

        self.indices = self._load_image_set_index() # a list of image indices

        if self.flip:
            self.indices = 2*self.indices
            #import pdb;pdb.set_trace()
            for i in range(int(len(self.indices)/2), len(self.indices)):
                self.indices[i] = self.indices[i] +'_flip'

        
        if self.shuffle_data:
            np.random.shuffle(self.indices)
        
        assert len(self.indices) > 0, "No training files"
        print("Number of files used: %s" % len(self.indices))
        
        img = self._load_file(0)
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]


    def _load_image_set_index(self):
        """
        Load a list of indexes listed in this dataset's image set file.
        Format: volname_sliceidx
        """
        image_index = []
        for name in self._vol_names:
            bbox_file = os.path.join(self.data_dir, 'seg_band_bbox',
                    name+'_bbox.txt')
            assert os.path.exists(bbox_file), \
                    'bbox path does not exist: {}'.format(bbox_file)
            with open(bbox_file) as fbbox:
                if self.use_empty:
                    vol_index = [x.strip().split(',')[0] for x in fbbox.readlines()]
                else:
                    vol_index = [x.strip().split(',')[0] for x in fbbox.readlines() if x.strip().split(',')[1]!='0']

            image_index += vol_index

        return image_index


    def _load_file(self, idx, dtype=np.float32, data_type='img'):
        img_flip = False
        if not self.flip:
            vol_name, sliceidx = self.indices[idx].rsplit('_',1)
        else:
            split_str = self.indices[idx].split('_')
            vol_name = split_str[0] + '_' + split_str[1]
            sliceidx = split_str[2]
            if len(split_str) > 3:
                # with flip
                img_flip = True

        sliceidx = int(sliceidx)
        #import pdb;pdb.set_trace()
        if data_type == 'img':
            im = np.array(self._image_h5f[vol_name][:,:,sliceidx],
                dtype=dtype)
            im -= self.mean
        else:
            im = np.array(self._label_h5f[vol_name+'_mask'][:,:,sliceidx],
                dtype=dtype)

        if img_flip:
            im = im[:, ::-1] # flip it 
        # stack to 3 channels
        #in_ = np.dstack((im, im, im))
        #in_ = in_.transpose((2,0,1))
        return im

    def _process_data(self, data):
        return data

    def _cylce_file(self):
        """ return the index of the file """
        self.file_idx += 1
        if self.file_idx >= len(self.indices):
            self.file_idx = 0 
            if self.shuffle_data:
                np.random.shuffle(self.indices)

    def _next_data(self):
        self._cylce_file()
        #image_name = self.data_files[self.file_idx]
        #label_name = image_name.replace(self.data_suffix, self.mask_suffix)

        img = self._load_file(self.file_idx, np.float32)
        label = self._load_file(self.file_idx, np.bool, 'label')
           # import pdb;pdb.set_trace()

        return img,label
