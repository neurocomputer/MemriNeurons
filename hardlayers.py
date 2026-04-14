"""
Функции выполнения матричного умножения, абстрактные
"""
import os
import csv
import copy
# import time
import pickle
import datetime

import numpy as np
from MemriNeurons.src import get_logger, change_log_file

# pylint: disable=C0301

class ElementWiseMatMulLayer():
    """
    Поэлементное умножение
    Пример:
    hardlayer = ElementWiseMatMulLayer(device)
    hardlayer.find_weights_model(layer_weights, layer_id='Dense_2')
    outputs = hardlayer.matmul(inputs)
    """

    logger = None

    def __init__(self, device, model_path):
        """
        device - устройство у которого есть методы
        multiply - умножение 2 чисел (x, scale_x, sign_x, scale_w, sign_w, wl, bl)
        """
        self.device = device
        self.all_mem_weights = []
        self.mem_weights_coordinates = []
        self.mem_weights_scales = []
        self.layer_weights = []
        self.model_path = model_path

    def update_logger(self):
        """
        Задать логирование в файл
        """
        if self.logger is None:
            self.logger = get_logger()
        else:
            change_log_file(self.logger)

    def find_weights_model(self, layer_weights, max_weight, save_folder='', print_flag=True, layer_id=''):
        """
        Найти веса для слоя
        layer_weights - веса слоя, список из двух элементов - веса и пороги
        max_weight - определяется из железа
        """
        self.all_mem_weights = self.device.read_mem_weights(save_folder=save_folder)
        counter_params = 0
        self.mem_weights_coordinates = []
        self.mem_weights_scales = []
        self.layer_weights = copy.deepcopy(layer_weights)
        # работаем с весами
        ann_weights = layer_weights[counter_params]
        counter_params += 1
        # 1. опредиляем скеил
        scale_w = max_weight / np.max(np.abs(ann_weights))
        if print_flag:
            print(f'scale_w {scale_w}')
        # 2. скейлим веса
        ann_weights_scaled = np.abs(copy.deepcopy(ann_weights)*scale_w)
        # 3. ищем ближайшие занчения
        hard_weights = []
        for i in range(ann_weights_scaled.shape[0]):
            temp_hard_weights = []
            for j in range(ann_weights_scaled.shape[1]):
                temp_w = np.abs(copy.deepcopy(self.all_mem_weights)-ann_weights_scaled[i][j])
                wl_bl = np.unravel_index(np.argmin(temp_w), temp_w.shape)
                if print_flag:
                    print(f'wl {wl_bl[1]} bl {wl_bl[0]} Wo = {ann_weights[i][j]} Ws = {ann_weights_scaled[i][j]} Wm = {self.all_mem_weights[int(wl_bl[0])][int(wl_bl[1])]} ')
                temp_hard_weights.append({'wl': wl_bl[1], 'bl': wl_bl[0]})
            hard_weights.append(temp_hard_weights)
        # работаем с порогами
        ann_biases = layer_weights[counter_params]
        # 1. опредиляем скеил
        if np.max(np.abs(ann_biases)) != 0:
            scale_b = max_weight / np.max(np.abs(ann_biases))
        else:
            scale_b = max_weight # todo: почему?
        if print_flag:
            print(f'scale_b {scale_b}')
        # 2. скейлим пороги
        ann_biases_scaled = np.abs(copy.deepcopy(ann_biases)*scale_b)
        # 3. ищем ближайшие занчения
        hard_biases = []
        # pylint: disable=C0200
        for i in range(len(ann_biases_scaled)):
            temp_b = np.abs(copy.deepcopy(self.all_mem_weights)-ann_biases_scaled[i])
            wl_bl = np.unravel_index(np.argmin(temp_b),temp_b.shape)
            if print_flag:
                print(f'wl {wl_bl[1]} bl {wl_bl[0]} Bo = {ann_biases[i]} Bs = {ann_biases_scaled[i]} Bm = {self.all_mem_weights[int(wl_bl[0])][int(wl_bl[1])]}')
            hard_biases.append({'wl': wl_bl[1], 'bl': wl_bl[0]})
        self.mem_weights_coordinates.append(np.transpose(hard_weights))
        self.mem_weights_coordinates.append(hard_biases)
        self.mem_weights_scales.append(scale_w)
        self.mem_weights_scales.append(scale_b)

        if save_folder:
            with open(os.path.join(save_folder,
                                   f'all_mem_weights_coordinates_{layer_id}.pkl'),
                                   'wb') as fp:
                pickle.dump([self.mem_weights_coordinates, self.mem_weights_scales], fp)

    def matmul(self, input_data, **kwargs):
        """
        Матричное умножение по элементно
        out_type=model - вернет модельные результаты, иначе с мемристоров
        """

        #создаем папку для сохранения результата
        now = datetime.datetime.now()
        formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        result_dir = os.path.join(self.model_path, f'experiment_{formatted_date}')
        os.mkdir(result_dir)

        #создаем файл для сохранения результата умножения
        now = datetime.datetime.now()
        formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        fname_mac_result = os.path.join(result_dir, f'mac_{formatted_date}.csv')
        with open(fname_mac_result,'w', newline='', encoding='utf-8') as file:
            file_wr = csv.writer(file, delimiter=",")
            file_wr.writerow(['timestemp','neur', 'syn', 'wl', 'bl', 'dac', 'adc', 'res', 'truth'])

        #создаем файл для сохранения результата общего
        now = datetime.datetime.now()
        formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        fname_io_result = os.path.join(result_dir, f'IO_{formatted_date}.csv')
        with open(fname_io_result,'w', newline='', encoding='utf-8') as file:
            file_wr = csv.writer(file, delimiter=",")
            file_wr.writerow(['datestamp', 'outputs', 'outputs_mem'])

        result_log = []

        counter_params = 0
        layer_weights = self.layer_weights[counter_params]
        hard_weights = self.mem_weights_coordinates[counter_params]
        scale_w = self.mem_weights_scales[counter_params]
        counter_params += 1
        layer_biases = self.layer_weights[counter_params]
        hard_biases = self.mem_weights_coordinates[counter_params]
        scale_b = self.mem_weights_scales[counter_params]

        all_neurons_model = [] # выход всех модельных нейронов
        all_neurons_mem = [] # выход всех мемристорных нейронов

        # start_time = time.time()
        for inputs in input_data:
            now = datetime.datetime.now()
            formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
            result_log.append(formatted_date)
            scale_x = float(np.max(np.abs(inputs)))
            inputs_mem = inputs
            inputs_mem = list(map(lambda x, n=scale_x: np.round(abs(x)/n*0.3*4096/5), inputs_mem))
            neurons_model = []
            neurons_mem = []
            # нейроны
            for neuron in range(layer_weights.shape[1]):
                # веса
                mac_model = 0
                mac_mem = 0
                for synapse in range(layer_weights.shape[0]):
                    mul_model = layer_weights[synapse][neuron] * inputs[synapse]
                    mac_model += mul_model
                    adc = 0
                    if layer_weights[synapse][neuron] != 0 and inputs[synapse] != 0:
                        # мэмристоры
                        wl = hard_weights[neuron][synapse]['wl']
                        bl = hard_weights[neuron][synapse]['bl']
                        # start_time = time.time()
                        sign_w = np.sign(layer_weights[synapse][neuron])
                        sign_x = np.sign(inputs[synapse])
                        mul_res, adc = self.device.multiply(inputs[synapse],
                                                       scale_x,
                                                       sign_x,
                                                       scale_w,
                                                       sign_w,
                                                       wl,
                                                       bl)
                        # print(time.time() - start_time)
                        mac_mem += mul_res
                        print(mul_model, mul_res)
                    # пишем результат mac
                    now = datetime.datetime.now()
                    with open(fname_mac_result,'a', newline='', encoding='utf-8') as file:
                        file_wr = csv.writer(file, delimiter=",")
                        file_wr.writerow([now.strftime("%d.%m.%Y_%H.%M.%S"),
                                          neuron,
                                          synapse,
                                          wl,
                                          bl,
                                          int(inputs_mem[synapse]),
                                          adc,
                                          mul_res,
                                          mul_model])
                # биасы
                mac_model += layer_biases[neuron]
                if layer_biases[neuron] != 0:
                    # мэмристоры
                    wl = hard_biases[neuron]['wl']
                    bl = hard_biases[neuron]['bl']
                    sign_b = np.sign(layer_biases[neuron])
                    bias_res, adc = self.device.multiply(1,
                                                    1,
                                                    1,
                                                    scale_b,
                                                    sign_b,
                                                    wl,
                                                    bl)
                    mac_mem += bias_res
                    print(layer_biases[neuron], bias_res)
                # пишем результат mac
                now = datetime.datetime.now()
                with open(fname_mac_result,'a', newline='', encoding='utf-8') as file:
                    file_wr = csv.writer(file, delimiter=",")
                    file_wr.writerow([now.strftime("%d.%m.%Y_%H.%M.%S"),
                                      neuron,
                                      'b',
                                      wl,
                                      bl,
                                      246,
                                      adc,
                                      mul_res,
                                      layer_biases[neuron]])
                neurons_model.append(mac_model)
                neurons_mem.append(mac_mem)
            all_neurons_model.append(neurons_model)
            all_neurons_mem.append(neurons_mem)
            # пишем в файл общий результат
            print(neurons_model)
            result_log.append(np.argmax(neurons_model))
            result_log.append(np.argmax(neurons_mem))
            with open(fname_io_result,'a', newline='', encoding='utf-8') as file:
                file_wr = csv.writer(file, delimiter=",")
                file_wr.writerow(result_log)
            result_log = []
        all_neurons_model = np.array(all_neurons_model)
        all_neurons_mem = np.array(all_neurons_mem)
        if 'out_type' in kwargs:
            if kwargs['out_type'] == 'model':
                return all_neurons_model
        #print(time.time() - start_time)
        return all_neurons_mem
