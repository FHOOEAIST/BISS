from .augument import random_noise, random_flip, random_brightness
import numpy as np
import tensorflow as tf
from copy import deepcopy
from random import sample
import glob, os


class BaseGenerator(tf.keras.utils.Sequence):
    def __init__(self, patches_path, patch_shape, batch_size=8, num_classes=2, shuffle=True):
        self.batch_size = batch_size
        
        # Set the target directory for x and y data
        self._path_X = patches_path+'images/'

        # Load filenames as indices with glob ....
        self.indices = self.__getFilePaths()
        self.last_indices = deepcopy(self.indices)

        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        self._batch_shape = list(patch_shape)
        self._batch_shape.insert(0,batch_size)
        self._batch_shape = tuple(self._batch_shape)

    # Reset the indices to first state
    def reset(self):
        self.indices = deepcopy(self.last_indices)
        self.on_epoch_end()

    # Pick a single target from indices
    def pick_target(self, target):
        self.last_indices = deepcopy(self.indices)
        self.indices = [ elem for elem in self.indices if target in elem]
        self.on_epoch_end()

    def __getFilePaths(self) -> list():
        file_list = []
        for file in glob.glob(self._path_X + '*.npy'):
            if os.name == 'nt':
                name_target = file.split('\\')[-1].split('.npy')[0]
            else:
                name_target = file.split('/')[-1].split('.npy')[0]
            file_list.append(name_target)
        return file_list

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X = self.__get_data(batch)
        return X
    
    def __get_data(self, batch):
        X = np.zeros(self._batch_shape)
        
        for i, id in enumerate(batch):
            X[i] = np.load(self._path_X + id + '.npy')

        return X

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)


class TestGenerator(BaseGenerator):
    def __init__(self, patches_path, patch_shape, batch_size=8, num_classes=2, shuffle=True):
        super(TestGenerator, self).__init__(patches_path, patch_shape, batch_size, num_classes, shuffle)


class TrainGenerator(BaseGenerator):
    def __init__(self, patches_path, patch_shape, batch_size=8, num_classes=2, shuffle=True, augument=True, augument_factor=2):

        # Set if augumentation should be used
        self.augument = augument
        self.augument_factor = augument_factor

        # Consider addition augumentation images for batch_size
        if self.augument:
            super(TrainGenerator, self).__init__(patches_path, patch_shape, batch_size // (augument_factor + 1), num_classes, shuffle)

            self._batch_shape = list(patch_shape)
            self._batch_shape.insert(0,batch_size)
            self._batch_shape = tuple(self._batch_shape)
        else:
            super(TrainGenerator, self).__init__(patches_path, patch_shape, batch_size, num_classes, shuffle)

        # Set the target directory for y data
        self.__path_y = patches_path+'vessels/'

    # Exclude target files from indices
    def mask_target(self, target):
        self.last_indices = deepcopy(self.indices)
        self.indices = [ elem for elem in self.indices if elem.find(target)]
        self.on_epoch_end()

    # Split validation set from actual DataGenerator
    def validation_split(self, val_split=0.2) -> tf.keras.utils.Sequence:
        val_gen = deepcopy(self)
        val_gen.augument = False # Do not augument validation images

        n = int(len(self.indices) * val_split)
        val_indices = sample(self.indices, n)

        self.indices = [ elem for elem in self.indices if elem not in val_indices]
        self.on_epoch_end()

        val_gen.indices = [ elem for elem in val_gen.indices if elem in val_indices]
        val_gen.on_epoch_end()
        return val_gen

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]

        X, y = self.__get_data(batch)
        return X, y

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __get_data(self, batch):

        X = np.zeros(self._batch_shape)
        y = np.zeros(self._batch_shape)

        # Load non augumented patches
        for i, id in enumerate(batch):
            img = np.load(self._path_X + id + '.npy')
            vessels = np.load(self.__path_y + id + '.npy')
            X[i] = img
            y[i] = vessels

        # Fill up rest of batch with randomly augumented patches
        if (self.augument):
            n = X.shape[0] - len(batch)
            rand_indices = sample(self.indices, n)
            for i,id in enumerate(rand_indices, start=len(batch) ):
                img = np.load(self._path_X + id + '.npy')
                vessels = np.load(self.__path_y + id + '.npy')
                img,vessels = random_flip(img,vessels,0.5)
                img = random_brightness(img,0.1,0.3)
                img = random_noise(img,0.01,0.5)

                X[i] = img
                y[i] = vessels

        return X, y