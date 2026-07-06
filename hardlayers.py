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
    Element wise matrix multiplication
    Example:
    hardlayer = ElementWiseMatMulLayer(device, layer_id='Dense_1', save_folder='model')
    hardlayer.find_weights_model(layer_weights, max_weight, )
    outputs = hardlayer.matmul(inputs)
    """

    logger = None
    MAX_VOLTAGE = 0.3 # maximum input inference voltage
    DAC_BITS = 12 # dacs bit depth
    VOL_REF_DAC = 5 # dacs referense voltage

    def __init__(self, device, layer_id, save_folder=os.getcwd()) -> None:
        """
        device - memristors device middelware
        layer_id - layer name
        save_folder - path to result folder
        """
        self.device = device
        self.layer_id = layer_id
        self.all_mem_weights = []
        self.mem_weights_coordinates = []
        self.mem_weights_scales = []
        self.layer_weights = []
        self.save_folder = save_folder

        # make result folder
        self.result_dir = os.path.join(self.save_folder, self.layer_id)
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def update_logger(self) -> None:
        """
        Logging into a file
        """
        if self.logger is None:
            self.logger = get_logger()
        else:
            change_log_file(self.logger)

    def find_weights_model(self, layer_weights, max_weight, print_flag=True) -> None:
        """
        Make correspondings (synapse, neuron) <-> (bl, wl)
        layer_weights - layer weights and biases - [weights, biases]
        max_weight - maximum allowable weight value (calculated from memristors resistances)
        print_flag - print to terminal
        """
        self.all_mem_weights = self.device.read_mem_weights(save_folder=self.save_folder)
        counter_params = 0
        self.mem_weights_coordinates = []
        self.mem_weights_scales = []
        self.layer_weights = copy.deepcopy(layer_weights)
        # weight processing
        ann_weights = layer_weights[counter_params]
        counter_params += 1
        # 1. calculate scale
        scale_w = np.max(np.abs(ann_weights)) / max_weight 
        if print_flag:
            print(f'scale_w {scale_w}')
        # 2. scale weights
        ann_weights_scaled = np.abs(copy.deepcopy(ann_weights)/scale_w)
        # 3. find correspondings
        hard_weights = []
        for i in range(ann_weights_scaled.shape[0]):
            temp_hard_weights = []
            for j in range(ann_weights_scaled.shape[1]):
                temp_w = np.abs(copy.deepcopy(self.all_mem_weights)-ann_weights_scaled[i][j])
                wl_bl = np.unravel_index(np.argmin(temp_w), temp_w.shape)
                if print_flag:
                    print(f'bl {wl_bl[0]} wl {wl_bl[1]} Wo = {ann_weights[i][j]} Ws = {ann_weights_scaled[i][j]} Wm = {self.all_mem_weights[int(wl_bl[0])][int(wl_bl[1])]} ')
                temp_hard_weights.append({'bl': wl_bl[0], 'wl': wl_bl[1]})
            hard_weights.append(temp_hard_weights)
        # biases processing
        ann_biases = layer_weights[counter_params]
        # 1. calculate scale
        if np.max(np.abs(ann_biases)) != 0:
            scale_b = np.max(np.abs(ann_biases)) / max_weight
        else:
            scale_b = 0 / max_weight # todo: почему?
        if print_flag:
            print(f'scale_b {scale_b}')
        # 2. scale biases
        ann_biases_scaled = np.abs(copy.deepcopy(ann_biases)/scale_b)
        # 3. find correspondings
        hard_biases = []
        # pylint: disable=C0200
        for i in range(len(ann_biases_scaled)):
            temp_b = np.abs(copy.deepcopy(self.all_mem_weights)-ann_biases_scaled[i])
            wl_bl = np.unravel_index(np.argmin(temp_b),temp_b.shape)
            if print_flag:
                print(f'bl {wl_bl[0]} wl {wl_bl[1]} Bo = {ann_biases[i]} Bs = {ann_biases_scaled[i]} Bm = {self.all_mem_weights[int(wl_bl[0])][int(wl_bl[1])]}')
            hard_biases.append({'bl': wl_bl[0], 'wl': wl_bl[1]})
        self.mem_weights_coordinates.append(np.transpose(hard_weights))
        self.mem_weights_coordinates.append(hard_biases)
        self.mem_weights_scales.append(scale_w)
        self.mem_weights_scales.append(scale_b)

        if self.save_folder:
            with open(os.path.join(self.save_folder,
                                   f'all_mem_weights_coordinates_{self.layer_id}.pkl'),
                                   'wb') as fp:
                pickle.dump([self.mem_weights_coordinates, self.mem_weights_scales], fp)

    def matmul(self, input_data, **kwargs) -> np.ndarray:
        """
        Element wise matrix multiplication
        input_data - batch of inputs
        out_type=model - model data, otherwise memristive data
        """

        # matmul results storage path
        now = datetime.datetime.now()
        formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        fname_mac_result = os.path.join(self.result_dir, f'mac_{formatted_date}.csv')
        with open(fname_mac_result,'w', newline='', encoding='utf-8') as file:
            file_wr = csv.writer(file, delimiter=",")
            file_wr.writerow(['timestemp', 'neur', 'syn', 'bl', 'wl', 'dac', 'adc', 'res', 'truth'])

        # network results storage path
        now = datetime.datetime.now()
        formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        fname_io_result = os.path.join(self.result_dir, f'IO_{formatted_date}.csv')
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

        all_neurons_model = [] # model neurons output
        all_neurons_mem = [] # memristive neurons output

        # start_time = time.time()
        for inputs in input_data:
            now = datetime.datetime.now()
            formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
            result_log.append(formatted_date)
            scale_x = np.max(np.abs(inputs))/0.3
            # inputs_mem = inputs
            # autoscaling
            # inputs_mem = list(map(lambda x, n=scale_x: np.round(abs(x)/n*self.MAX_VOLTAGE*2**self.DAC_BITS/self.VOL_REF_DAC), inputs_mem))
            neurons_model = []
            neurons_mem = []
            # neurons
            for neuron in range(layer_weights.shape[1]):
                # synapses
                mac_model = 0
                mac_mem = 0
                for synapse in range(layer_weights.shape[0]):
                    mul_model = layer_weights[synapse][neuron] * inputs[synapse]
                    mac_model += mul_model
                    adc = 0
                    mul_res = 0
                    # memristors
                    wl = hard_weights[neuron][synapse]['wl']
                    bl = hard_weights[neuron][synapse]['bl']
                    if layer_weights[synapse][neuron] != 0 and inputs[synapse] != 0:
                        # start_time = time.time()
                        sign_w = np.sign(layer_weights[synapse][neuron])
                        # sign_x = np.sign(inputs[synapse])
                        # multiplication - multiplication 2 numbers (x, bl, wl, scale_x=1, scale_w=1, sign_w=1)
                        # print(inputs[synapse], scale_x, scale_w)
                        mul_res, adc = self.device.multiplication(inputs[synapse],
                                                                  bl,
                                                                  wl,
                                                                  scale_x,
                                                                  scale_w,
                                                                  sign_w)
                        # print(time.time() - start_time)
                        mac_mem += mul_res
                        # print(mul_model, mul_res)
                    # mac operation saving
                    now = datetime.datetime.now()
                    with open(fname_mac_result,'a', newline='', encoding='utf-8') as file:
                        file_wr = csv.writer(file, delimiter=",")
                        file_wr.writerow([now.strftime("%d.%m.%Y_%H.%M.%S"),
                                          neuron,
                                          synapse,
                                          bl,
                                          wl,
                                          inputs[synapse]/scale_x,
                                          adc,
                                          mul_res,
                                          mul_model])
                # biases
                mac_model += layer_biases[neuron]
                adc = 0
                bias_res = 0
                if layer_biases[neuron] != 0:
                    # memristors
                    wl = hard_biases[neuron]['wl']
                    bl = hard_biases[neuron]['bl']
                    sign_b = np.sign(layer_biases[neuron])
                    bias_res, adc = self.device.multiplication(1,
                                                               bl,
                                                               wl,
                                                               1/0.3,
                                                               scale_b,
                                                               sign_b)
                    mac_mem += bias_res
                    # print(layer_biases[neuron], bias_res)
                # mac operation saving
                now = datetime.datetime.now()
                with open(fname_mac_result,'a', newline='', encoding='utf-8') as file:
                    file_wr = csv.writer(file, delimiter=",")
                    file_wr.writerow([now.strftime("%d.%m.%Y_%H.%M.%S"),
                                      neuron,
                                      'b',
                                      bl,
                                      wl,
                                      1/0.3, # todo: исправить хардкод
                                      adc,
                                      bias_res,
                                      layer_biases[neuron]])
                neurons_model.append(mac_model)
                neurons_mem.append(mac_mem)
            all_neurons_model.append(neurons_model)
            all_neurons_mem.append(neurons_mem)
            # neurons operation saving
            # print(neurons_model)
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
        # print(time.time() - start_time)
        return all_neurons_mem

class SingleCrossbarMatMulLayer():
    """
    Matrix multiplication using single crossbar
    """

    logger = None

    def __init__(self, device, layer_id, save_folder=os.getcwd()):
        self.device = device
        self.layer_id = layer_id
        self.save_folder = save_folder

        # make result folder
        self.result_dir = os.path.join(self.save_folder, self.layer_id)
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

    def update_logger(self):
        """
        Logging into a file
        """
        if self.logger is None:
            self.logger = get_logger()
        else:
            change_log_file(self.logger)

    def matmul(self, input_data, **kwargs):
        """
        
        """

        # matmul results storage path
        now = datetime.datetime.now()
        formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        fname_mac_result = os.path.join(self.result_dir, f'mac_{formatted_date}.csv')
        with open(fname_mac_result,'w', newline='', encoding='utf-8') as file:
            file_wr = csv.writer(file, delimiter=",")
            file_wr.writerow(['timestemp', 'neur', 'syn', 'bl', 'wl', 'dac', 'adc', 'res', 'truth'])

        # network results storage path
        now = datetime.datetime.now()
        formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        fname_io_result = os.path.join(self.result_dir, f'IO_{formatted_date}.csv')
        with open(fname_io_result,'w', newline='', encoding='utf-8') as file:
            file_wr = csv.writer(file, delimiter=",")
            file_wr.writerow(['datestamp', 'outputs', 'outputs_mem'])

        result_log = []

        for inputs in input_data:
            now = datetime.datetime.now()
            formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
            result_log.append(formatted_date)
            scale_x = np.max(np.abs(inputs))
            inputs_mem = inputs
            # autoscaling
            inputs_mem = list(map(lambda x, n=scale_x: np.round(abs(x)/n*self.MAX_VOLTAGE*2**self.DAC_BITS/self.VOL_REF_DAC), inputs_mem))
            neurons_model = []
            neurons_mem = []

        for i in range(self.parent.input_array_scaled.shape[0]): #100
            # подготавливаем семпл
            # v_dac = [0 for i in range(32)] # todo: перенести в драйвер
            v_dac = [0 for i in range(self.parent.input_array_scaled.shape[1])]
            for h in range(self.parent.input_array_scaled.shape[1]):
                if self.parent.input_array_scaled[i][h] > 0.3:
                    v_dac[h] = v2d(self.parent.parent.man.dac_bit,
                                self.parent.parent.man.VOL_REF_DAC,
                                0.3)
                else:
                    v_dac[h] = v2d(self.parent.parent.man.dac_bit,
                                self.parent.parent.man.VOL_REF_DAC,
                                self.parent.input_array_scaled[i][h])
            # проходим по всем строкам кроссбара
            for j in range(self.parent.parent.man.col_num): #8
                if self.parent.matmul_predicted_results[i][j] < self.parent.vol_comp:
                    # маскирование v_adc
                    v_dac_current = deepcopy(v_dac)
                    # наложение на v dac 8-ми разных масок
                    print(v_dac_current)
                    for z in range(self.parent.parent.man.row_num):
                        if self.mask[j][z] == 0:
                            v_dac_current[z] = 0
                    task = {'mode_flag': 10,
                            'vol': v_dac_current,
                            'id': 0,
                            'wl': j}
                    v_adc, _ = self.parent.parent.man.conn.impact(task)
                    #print(v_adc)
                else:
                    v_adc = 0
                self.parent.matmul_crossbar_results[i][j] = a2v(self.parent.parent.man.gain,
                                        self.parent.parent.man.adc_bit,
                                        self.parent.parent.man.vol_ref_adc,
                                        v_adc)
                counter += 1
                self.count_changed.emit(counter)
                self.value_got.emit(v_adc)
        self.progress_finished.emit(counter)

class NPULayer():
    """
    Слой нейронки на базе НПУ
    """

    logger = None

    def __init__(self):
        pass

    
    def config(self, config_path):
        """
        Конфигурация проца
        """
        self.config_path = config_path
        try:
            with open(os.path.join(config_path, 'npu_config.json'), "r", encoding="utf-8") as f:
                self.cores_configs = json.load(f)
        except FileNotFoundError:
            print('Нет файла npu_config.json')
        except json.decoder.JSONDecodeError:
            print('Ошибка чтения файла npu_config.json')
        else:
            self.config_name = self.cores_configs['name']
            for core_id in self.cores_configs['cores']:
                if not self.silent:
                    print(f'Конфигурация ядра {core_id}')
                self.cores[core_id].config(number_accum = self.cores_configs['cores'][core_id]['number_accum'],
                                            number_sum = self.cores_configs['cores'][core_id]['number_sum'],
                                            number_input = self.cores_configs['cores'][core_id]['number_input'],
                                            number_repeat = self.cores_configs['cores'][core_id]['number_repeat'],
                                            mode_summator = self.cores_configs['cores'][core_id]['mode_summator'],
                                            mode_input = self.cores_configs['cores'][core_id]['mode_input'],
                                            steps_col = self.cores_configs['cores'][core_id]['steps_col'],
                                            steps_line = self.cores_configs['cores'][core_id]['steps_line']
                                            )
                if not self.silent:
                    self.cores[core_id].info("config")
                    print()
