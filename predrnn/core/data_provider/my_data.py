__author__ = 'gaozhifeng'
import numpy as np
import os
import cv2
from PIL import Image
import logging
import random

logger = logging.getLogger(__name__)


class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, 1)).astype(
            self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice

        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))


class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.category_1 = ['/extend/my_data/train_data']
        self.category_2 = ['/extend/my_data_test/test_data']
        self.category = self.category_1 + self.category_2
        self.image_width = input_param['image_width']

        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    def load_data(self, paths, mode='train'):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        path = paths[0]
        if mode == 'train':
            path = self.category_1[0]
            seq_num = 50
        elif mode == 'test':
            path = self.category_2[0]
            seq_num = 9
        else:
            print("ERROR!")
        print('begin load data' + str(path))

        frames_np = []
        files = sorted(os.listdir(path))
        # print(files)
        # exit()
        for i in range(seq_num):
            file_list = files[i*20:i*20+20]
            print("file_list", i, file_list)
            file_list = [os.path.join(path,f) for f in file_list]
            for file in file_list:
                frame_im = np.fromfile(file,dtype=np.uint8).reshape(700,900)
                frame_im[frame_im >= 80] = 0
                frame_np = np.array(frame_im)  # (1000, 1000) numpy array
                # print(frame_np.shape)
                # frame_np_list.append(frame_np)
                frames_np.append(frame_np)
        indices = [i*20 for i in range(seq_num)]
        frames_np = np.asarray(frames_np)
        data = np.zeros((frames_np.shape[0], self.image_width, self.image_width, 1))
        for i in range(len(frames_np)):
            temp = np.float32(frames_np[i, :, :])
            data[i, :, :, 0] = cv2.resize(temp, (self.image_width, self.image_width)) / 255
        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)

