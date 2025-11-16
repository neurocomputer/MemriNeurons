"""
Компоненты модели без спец фреймворка
"""

import math
import copy
import numpy as np

def feature_map_reshape(feature_map, kernel_shape, strides_shape):
    """
    Разрезатель одной карты признаков
    feature_map.shape = (a, b, c, d)
    a - кол-во семплов
    b - высота 
    c - ширина
    d - кол-во каналов
    crop.shape = (e, f, g)
    # e - кол-во семплов
    # f - кол-во порций данных в семпле
    # g - развернуто под ядро
    """
    size = (feature_map.shape[1], feature_map.shape[2])
    conv_x = kernel_shape[0]
    stride_x = strides_shape[0]
    conv_y = kernel_shape[1]
    stride_y = strides_shape[1]
    channels = kernel_shape[2] # кол-во каналов

    scale_factor_x = math.ceil((size[0] - conv_x)/stride_x + 1)
    scale_factor_y = math.ceil((size[1] - conv_y)/stride_y + 1)

    crop = []
    for sample in feature_map:
        sample_cutted = []
        for i in range(scale_factor_x):
            for j in range(scale_factor_y):
                area = np.zeros(shape=(stride_x, stride_y, channels))
                x1 = i*conv_x
                x2 = i*conv_x+stride_x
                if x2 > size[0]:
                    x2 = size[0]
                y1 = j*conv_y
                y2 = j*conv_y+stride_y
                if y2 > size[1]:
                    y2 = size[1]
                croped_image = sample[x1:x2,y1:y2]
                area[0:croped_image.shape[0], 0:croped_image.shape[1]] = croped_image
                sample_cutted.append(area.flatten())
        crop.append(np.array(sample_cutted))

    return np.array(crop), scale_factor_x, scale_factor_y

class Sequential():
    """
    Модель Sequential
    """

    layers = [] # слои

    def add(self, layer):
        """
        Добавить слой
        """
        self.layers.append(layer)

    def predict(self, input_data):
        """
        Инференс модели
        """
        for _, layer in enumerate(self.layers):
            output_data = layer(input_data)
            input_data = output_data
        return output_data

class Activations():
    """
    Функции активации
    """

    def linear(self, x):
        """
        Линейная функция
        """
        return x

    def relu(self, x):
        """
        ReLu
        """
        x[x < 0] = 0
        return x

    def softmax(self, x, axis=-1):
        """
        softmax
        """
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)

        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    def sigmoid(self, x):
        """
        Sigmoid
        """
        return np.where(x >= 0,
                   1 / (1 + np.exp(-x)),
                   np.exp(x) / (1 + np.exp(x)))

class Layer():
    """
    Кастомный слой
    """

    activations = Activations()
    activation = None
    weights = None
    biases = None

    def __init__(self, **kwargs):
        """
        Инициализация слоя
        """
        if 'activation' in kwargs:
            assert isinstance(kwargs['activation'], str)
            if kwargs['activation'] in dir(self.activations):
                self.activation = getattr(self.activations, kwargs['activation'])
        else:
            self.activation = self.activations.linear
        if 'weights' in kwargs:
            assert isinstance(kwargs['weights'], list)
            self.set_weights(kwargs['weights'][0])
            if len(kwargs['weights']) > 1:
                self.set_biases(kwargs['weights'][1])

    def set_weights(self, weights):
        """
        Установить веса в модель
        """
        self.weights = copy.deepcopy(weights)

    def set_biases(self, biases):
        """
        Установить пороги в модель
        """
        self.biases = copy.deepcopy(biases)

    def get_weights(self):
        """
        Вернуть значения весов и порогов
        """
        model_params = []
        if not self.weights is None:
            model_params.append(copy.deepcopy(self.weights))
        if not self.biases is None:
            model_params.append(copy.deepcopy(self.biases))
        return model_params

    @staticmethod
    def matmul(inputs, weights):
        """
        Матричное умножение (можно переопределять)
        """
        # print(inputs.shape, weights[0].shape, weights[1].shape)
        return inputs @ weights[0] + weights[1]

class Dense(Layer):
    """
    Полносвязный слой
    """

    def __call__(self, input_data):
        """
        Инференс
        input_data - может быть сразу несколько семплов, c
        weights.shape = (a, b)
        input_data.shape = (c, a)
        output_data.shape = (c, b)
        """
        # получаем матрицу весов
        weights = self.get_weights()[0]
        # получаем пороги (если они есть)
        if len(self.get_weights()) > 1:
            biases = self.get_weights()[1]
        else:
            biases = np.zeros(shape=(weights.shape[1],))
            # print(biases)
        # выход слоя
        output_data = self.matmul(input_data, [weights, biases])
        output_data = self.activation(output_data)
        return output_data

class Conv2D(Layer):
    """
    Сверточный слой
    """

    strides = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'strides' in kwargs:
            assert isinstance(kwargs['strides'], tuple)
            self.strides = kwargs['strides']

    def __call__(self, input_data):
        """
        Инференс
        weights.shape = (a, b, c, d)
        input_data.shape = (e, f, g, h)
        output_data.shape = (e, i, j, d)
        a, b - размер ядра
        c - кол-во каналов ядра
        d - кол-во ядер
        e - кол-во семплов
        f - высота
        g - ширина
        h - кол-во каналов
        i - math.ceil(f/a)
        j - math.ceil(g/b)
        """
        # получаем матрицу весов
        kernels = self.get_weights()[0]
        # получаем пороги (если они есть)
        if len(self.get_weights()) > 1:
            biases = self.get_weights()[1]
        else:
            biases = np.zeros(shape=(kernels.shape[3],))
            # print(biases)
        # модернизируем вход
        crop, sc_x, sc_y = feature_map_reshape(input_data, kernels.shape, self.strides)
        # выход слоя
        output_data = []
        for img_item in crop:
            for item in img_item:
                for i in range(kernels.shape[3]): # цикл по сверткам
                    kernel = kernels[:,:,:,i]
                    out = self.matmul(item, [kernel.flatten(), biases[i]])
                    output_data.append(out)
        output_data = np.array(output_data).reshape((len(crop), sc_x, sc_y, kernels.shape[3]))
        output_data = self.activation(output_data)
        return output_data

class Flatten(Layer):
    """
    Слой Flatten
    """

    def __call__(self, input_data):
        """
        Инференс
        """
        if len(input_data.shape) == 1:
            # если массив одномерный, добавляем batch размерность
            return input_data.reshape(1, -1)
        else:
            batch_size = input_data.shape[0]
            return input_data.reshape(batch_size, -1)
