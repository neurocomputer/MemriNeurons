import os
import csv
import copy
import time
import pickle
import datetime

import numpy as np
from gui.src import choose_cells

def convert_res_to_weight(res: int) -> float: # todo: переписать на вызов функции a2r
    """
    Конвертер сопротивления в вес
    """
    weight = 3000/(3000 + res)
    return weight

def a2v(adc_value:int):
    _vol_ref_adc = 5
    _adc_bit = 14
    _gain = 11.11
    vol_value = round(((adc_value * _vol_ref_adc)/((2 ** _adc_bit) - 1)) / _gain, 5)
    return vol_value

def softmax(vec):
    exponential = np.exp(vec)
    probabilities = exponential / np.sum(exponential)
    return probabilities

class ElementWiseMatMul():

    def __init__(self, model_path, conn):
        
        self.model_path = model_path
        self.conn = conn
        self.all_mem_weights = None
        self.mem_weights_coordinates = []
        self.mem_weights_scales = []
        self.weights = None

    def read_mem_weights(self, save_file=True):
        """
        Прочитать все веса в кроссбаре
        """
        # дополнительные ограничения
        dop_mod=0
        # масштабирование веса
        weight_correction = 1
        # ограничить диапазон
        diap = 0
        diap_min = 100 # Ohm
        diap_max = 100000 # Ohm
        # ограничить работоспособными
        cells_filter = 0
        if cells_filter:
            cells, _ = choose_cells('good_cells.csv', 8, 32)
        # размер кроссбара
        wl_all = 8
        bl_all = 32
        # прочитать все веса в сети
        all_mem_weights = np.zeros(shape=(bl_all, wl_all), dtype=float)
        for wl in range(wl_all):
            for bl in range(bl_all):
                adc = self.conn.mode_7(0,0,0,0,1,wl,bl)
                # повторный запрос
                if adc[0] < 10:
                    adc = self.conn.mode_7(0,0,0,0,1,wl,bl)
                adc_value = adc[0]
                if adc_value < 50:
                    adc_value = 50
                res = (11.0*3000*0.3*(2**14))/(adc_value*5)-10-3000 # todo: переписать на вызов функции a2r
                if res <= 0:
                    res = 0.00000001
                if dop_mod:
                    if diap:
                        if diap_min > res or res > diap_max:
                            all_mem_weights[bl][wl] = 1000000
                        else:
                            print(f'wl {wl} bl {bl} = {int(res)}')
                            all_mem_weights[bl][wl] = convert_res_to_weight(int(res)) # todo: переписать на вызов функции a2r
                    if cells_filter:
                        if not (wl, bl) in cells:
                            all_mem_weights[bl][wl] = 1000000
                        else:
                            print(f'wl {wl} bl {bl} = {int(res)}')
                            all_mem_weights[bl][wl] = convert_res_to_weight(int(res))
                else:
                    print(f'wl {wl} bl {bl} = {int(res)}, {convert_res_to_weight(int(res))*weight_correction}')
                    all_mem_weights[bl][wl] = convert_res_to_weight(int(res))*weight_correction
        self.all_mem_weights = all_mem_weights
        if save_file:
            if not os.path.exists(self.model_path): os.mkdir(self.model_path)
            with open(os.path.join(self.model_path, 'all_mem_weights.pkl'), 'wb') as fp:
                pickle.dump(all_mem_weights, fp)

    def find_weights_model(self, weights, save_file=True, layer_id=''):
        """
        Найти веса для слоя
        """
        num_layers = 1
        counter_params = 0

        self.mem_weights_coordinates = []
        self.mem_weights_scales = []
        self.weights = weights

        for i in range(num_layers):
            # работаем с весами
            ann_weights = weights[counter_params]
            counter_params += 1
            #1. опредиляем скеил
            scale_w = 0.96 / np.max(np.abs(ann_weights))
            print(f'scale_w {scale_w}')
            #2 скейлм веса
            ann_weights_scaled = np.abs(copy.deepcopy(ann_weights)*scale_w)
            #3 ищим ближайшие занчения
            HARD_WEIGHTS = []
            for i in range(ann_weights_scaled.shape[0]):
                temp_HARD_WEIGHTS = [] 
                for j in range(ann_weights_scaled.shape[1]):
                    temp_w = np.abs(copy.deepcopy(self.all_mem_weights)-ann_weights_scaled[i][j])         
                    wl_bl = np.unravel_index(np.argmin(temp_w),temp_w.shape)
                    print(f'wl {wl_bl[1]} bl {wl_bl[0]} Wo = {ann_weights[i][j]} Ws = {ann_weights_scaled[i][j]} Wm = {self.all_mem_weights[wl_bl[0]][wl_bl[1]]} ')
                    temp_HARD_WEIGHTS.append({'wl': wl_bl[1], 'bl': wl_bl[0]})
                HARD_WEIGHTS.append(temp_HARD_WEIGHTS)
            # работаем с порогами
            ann_biases = weights[counter_params]
            counter_params += 1
            #1. опредиляем скеил
            if np.max(np.abs(ann_biases)) != 0:
                scale_b = 0.96 / np.max(np.abs(ann_biases))
            else:
                scale_b = 1
            print(f'scale_b {scale_b}')
            #2 скейлм пороги
            ann_biases_scaled = np.abs(copy.deepcopy(ann_biases)*scale_b)
            #3 ищим ближайшие занчения
            HARD_BIASES = []
            for i in range(len(ann_biases_scaled)):
                temp_b = np.abs(copy.deepcopy(self.all_mem_weights)-ann_biases_scaled[i])
                wl_bl = np.unravel_index(np.argmin(temp_b),temp_b.shape)
                print(f'wl {wl_bl[1]} bl {wl_bl[0]} Bo = {ann_biases[i]} Bs = {ann_biases_scaled[i]} Bm = {self.all_mem_weights[wl_bl[0]][wl_bl[1]]}')
                HARD_BIASES.append({'wl': wl_bl[1], 'bl': wl_bl[0]})
            self.mem_weights_coordinates.append(np.transpose(HARD_WEIGHTS))
            self.mem_weights_coordinates.append(HARD_BIASES)
            self.mem_weights_scales.append(scale_w)
            self.mem_weights_scales.append(scale_b)

        with open(os.path.join(self.model_path, f'all_mem_weights_coordinates{layer_id}.pkl'), 'wb') as fp:
            pickle.dump([self.mem_weights_coordinates, self.mem_weights_scales], fp)

    def process_layer(self, input_data):
        """
        Работа слоя
        """

        # создаем папку для сохранения результата
        # now = datetime.datetime.now()
        # formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        # result_dir = os.path.join(self.model_path, f'experiment_{formatted_date}')
        # os.mkdir(result_dir)

        # создаем файл для сохранения результата умножения
        # now = datetime.datetime.now()
        # formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        # fname_mac_result = os.path.join(result_dir, f'mac_{formatted_date}.csv')
        # with open(fname_mac_result,'w', newline='', encoding='utf-8') as file:
        #     file_wr = csv.writer(file, delimiter=",")
        #     file_wr.writerow(['neur', 'syn', 'wl', 'bl', 'dac', 'adc', 'res', 'truth'])

        # создаем файл для сохранения результата общего
        # now = datetime.datetime.now()
        # formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        # fname_io_result = os.path.join(result_dir, f'IO_{formatted_date}.csv')
        # with open(fname_io_result,'w', newline='', encoding='utf-8') as file:
        #     file_wr = csv.writer(file, delimiter=",")
        #     file_wr.writerow(['datestamp', 'outputs', 'outputs_mem'])

        # result_log = []

        counter_params = 0
        layer_weights = self.weights[counter_params]
        HARD_WEIGHTS = self.mem_weights_coordinates[counter_params]
        SCALE_W = self.mem_weights_scales[counter_params]
        counter_params += 1
        layer_biases = self.weights[counter_params]
        HARD_BIASES = self.mem_weights_coordinates[counter_params]
        SCALE_B = self.mem_weights_scales[counter_params]

        all_neurons_mem = []
        all_neurons_model = []
        start_time = time.time()
        for inputs in input_data: 
            # now = datetime.datetime.now()
            # formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
            # result_log.append(formatted_date)
            scale_x = np.max(np.abs(inputs))
            inputs_mem = inputs
            inputs_mem = list(map(lambda x: np.round(abs(x)/scale_x*0.3*4096/5), inputs_mem))
            neurons_model = []
            neurons_mem = []
            # нейроны
            for neuron in range(layer_weights.shape[1]):
                # веса
                # mac_model = 0
                mac_mem = 0
                for synapse in range(layer_weights.shape[0]):
                    # mul_model = layer_weights[synapse][neuron] * inputs[synapse]
                    # mac_model += mul_model
                    if layer_weights[synapse][neuron] != 0:
                        # мэмристоры
                        wl = HARD_WEIGHTS[neuron][synapse]['wl']
                        bl = HARD_WEIGHTS[neuron][synapse]['bl']
                        #start_time = time.time()
                        res = self.conn.mode_9(int(inputs_mem[synapse]), 0, wl, bl)[0]
                        #print(time.time() - start_time)
                        mul = a2v(res)
                        sign_w = np.sign(layer_weights[synapse][neuron])
                        sign_i = np.sign(inputs[synapse])
                        mul_res = mul * sign_i * sign_w / SCALE_W / 0.3 * scale_x
                        mac_mem += mul_res
                    # пишем результат mac
                    # with open(fname_mac_result,'a', newline='', encoding='utf-8') as file:
                    #     file_wr = csv.writer(file, delimiter=",")
                    #     file_wr.writerow([neuron, synapse, wl, bl, int(inputs_mem[synapse]), res, mul_res, mul_model])
                # биасы
                # mac_model += layer_biases[neuron]
                if layer_biases[neuron] != 0:
                    # мэмристоры
                    wl = HARD_BIASES[neuron]['wl']
                    bl = HARD_BIASES[neuron]['bl']
                    res = self.conn.mode_9(246, 0, wl, bl)[0]
                    mul = a2v(res)
                    sign = np.sign(layer_biases[neuron])
                    mul_res = mul * sign / SCALE_B / 0.3
                    mac_mem += mul_res
                # пишем результат mac
                # with open(fname_mac_result,'a', newline='', encoding='utf-8') as file:
                #     file_wr = csv.writer(file, delimiter=",")
                #     file_wr.writerow([neuron, 'b', wl, bl, 246, res, mul_res, layer_biases[neuron]])
                # neurons_model.append(mac_model)
                neurons_mem.append(mac_mem)
            
            # all_neurons_model.append(neurons_model)
            all_neurons_mem.append(neurons_mem)

            # пишем в файл общий результат
            # result_log.append(np.argmax(softmax(neurons_model)))
            # result_log.append(np.argmax(softmax(neurons_mem)))
            # with open(fname_io_result,'a', newline='', encoding='utf-8') as file:
            #     file_wr = csv.writer(file, delimiter=",")
            #     file_wr.writerow(result_log)

        all_neurons_mem = np.array(all_neurons_mem)
        #print(time.time() - start_time)
        return all_neurons_mem
