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
        self.all_mem_weights = [] # all memristive weights
        self.mem_weights_coordinates = [] # [weights, biases] of {'bl', 'wl'}
        self.mem_weights_scales = [] # scale_w, scale_b
        self.layer_weights = [] # source layer weights [weights, biases]
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
    Only positive voltages can be applied to BL, so signed inputs and weights are split:
    X = X+ - X-, W = (G+ - G-)/scale_w, every neuron uses two columns (G+, G-)
    X*W = (X+*G+ - X-*G+ - X+*G- + X-*G-)/scale_w
    Example:
    hardlayer = SingleCrossbarMatMulLayer(device, layer_id='Dense_1', save_folder='model')
    hardlayer.write_weights(weights)
    outputs = hardlayer.matmul(inputs)
    """

    logger = None
    G_MIN = 0.0033 # minimum conductance (mvm weight) of a memristor
    G_MAX = 0.05 # maximum conductance (mvm weight) of a memristor

    def __init__(self, device, layer_id, save_folder=os.getcwd()) -> None:
        """
        device - memristors device middelware
        layer_id - layer name
        save_folder - path to result folder
        """
        self.device = device
        self.layer_id = layer_id
        self.save_folder = save_folder

        self.layer_weights = None # source layer weights (inputs, neurons)
        self.target_conductances = None # target conductances (inputs, 2*neurons): G+, G- columns
        self.mem_conductances = None # written conductances (inputs, 2*neurons)
        self.scale_w = 1 # G = W*scale_w
        self.left_top = (0, 0) # (bl, wl) of the first used cell

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

    def map_weights(self, weights) -> np.ndarray:
        """
        Map weights to conductances pairs
        weights - (inputs, neurons)
        W = 0 -> G+ = G- = G_MIN, |W| = max|W| -> |G+ - G-| = G_MAX - G_MIN
        return conductances (inputs, 2*neurons), columns 2j - G+, 2j+1 - G-
        """
        weights = np.asarray(weights, dtype=float)
        max_weight = np.max(np.abs(weights))
        self.scale_w = (self.G_MAX - self.G_MIN) / max_weight if max_weight != 0 else 1
        conductances = np.full((weights.shape[0], 2*weights.shape[1]), self.G_MIN, dtype=float)
        conductances[:, 0::2] += np.where(weights > 0, weights, 0) * self.scale_w
        conductances[:, 1::2] += np.where(weights < 0, -weights, 0) * self.scale_w
        return conductances

    def write_weights(self, weights, left_top=None, tolerance=0.02, attempts=5, silent=True) -> np.ndarray:
        """
        Writing weights into the crossbar starting from left_top
        weights - (inputs, neurons)
        left_top - (bl, wl) of the first cell
        tolerance - allowable write error relative to (G_MAX - G_MIN)
        attempts - number of cell rewrites while error > tolerance
        return written weights (G+ - G-)/scale_w
        """
        if left_top is None:
            left_top = (0, 0)
        weights = np.asarray(weights, dtype=float)
        assert weights.ndim == 2
        assert left_top[0] + weights.shape[0] <= self.device.ROW_NUM
        assert left_top[1] + 2*weights.shape[1] <= self.device.COL_NUM
        self.layer_weights = copy.deepcopy(weights)
        self.left_top = left_top
        self.target_conductances = self.map_weights(weights)
        if not silent:
            print(f'scale_w {self.scale_w}')
        self.device.set_mvm()
        self.mem_conductances = np.zeros_like(self.target_conductances)
        for i in range(self.target_conductances.shape[0]):
            for j in range(self.target_conductances.shape[1]):
                bl = left_top[0] + i
                wl = left_top[1] + j
                for _ in range(attempts):
                    self.mem_conductances[i][j], _, _ = self.device.write_weight(bl, wl, self.target_conductances[i][j])
                    if abs(self.mem_conductances[i][j] - self.target_conductances[i][j]) <= tolerance*(self.G_MAX - self.G_MIN):
                        break
                if not silent:
                    print(f'bl {bl} wl {wl} Gt = {self.target_conductances[i][j]} Gm = {self.mem_conductances[i][j]}')
        if self.save_folder:
            with open(os.path.join(self.save_folder,
                                   f'single_crossbar_weights_{self.layer_id}.pkl'),
                                   'wb') as fp:
                pickle.dump([self.layer_weights,
                             self.target_conductances,
                             self.mem_conductances,
                             self.scale_w,
                             self.left_top], fp)
        return self.get_mem_weights()

    def get_mem_weights(self) -> np.ndarray:
        """
        Weights calculated from written conductances (inputs, neurons)
        """
        return (self.mem_conductances[:, 0::2] - self.mem_conductances[:, 1::2]) / self.scale_w

    def matmul(self, input_data, **kwargs) -> np.ndarray:
        """
        Matrix multiplication in the crossbar
        input_data - batch of inputs (samples, inputs) or single input (inputs, )
        out_type=model - model data, otherwise memristive data
        return (samples, neurons)
        """
        input_data = np.asarray(input_data, dtype=float)
        single = input_data.ndim == 1
        input_data = np.atleast_2d(input_data)
        assert input_data.shape[1] == self.layer_weights.shape[0]

        if kwargs.get('out_type') == 'model':
            result = input_data @ self.layer_weights
            return result[0] if single else result

        # matmul results storage path
        now = datetime.datetime.now()
        formatted_date = now.strftime("%d.%m.%Y_%H.%M.%S")
        fname_mac_result = os.path.join(self.result_dir, f'mac_{formatted_date}.csv')
        with open(fname_mac_result,'w', newline='', encoding='utf-8') as file:
            file_wr = csv.writer(file, delimiter=",")
            file_wr.writerow(['timestemp', 'sample', 'neur', 'wl_plus', 'wl_minus',
                              'xp_gp', 'xm_gp', 'xp_gm', 'xm_gm', 'scale_x', 'res', 'truth'])

        self.device.set_mvm()
        bl_start, wl_start = self.left_top
        all_neurons_mem = []
        for sample, inputs in enumerate(input_data):
            max_input = np.max(np.abs(inputs))
            scale_x = max_input / self.device.MAX_VOL_INFERENCE if max_input != 0 else 1
            # place inputs on BL from left_top, voltages <= MAX_VOL_INFERENCE
            x_plus = np.zeros(self.device.ROW_NUM)
            x_minus = np.zeros(self.device.ROW_NUM)
            x_plus[bl_start:bl_start+len(inputs)] = np.where(inputs > 0, inputs, 0) / scale_x
            x_minus[bl_start:bl_start+len(inputs)] = np.where(inputs < 0, -inputs, 0) / scale_x
            neurons_mem = []
            for neuron in range(self.layer_weights.shape[1]):
                wl_plus = wl_start + 2*neuron
                wl_minus = wl_plus + 1
                xp_gp = self._dot_product(x_plus, wl_plus)
                xm_gp = self._dot_product(x_minus, wl_plus)
                xp_gm = self._dot_product(x_plus, wl_minus)
                xm_gm = self._dot_product(x_minus, wl_minus)
                mac_mem = (xp_gp - xm_gp - xp_gm + xm_gm) * scale_x / self.scale_w
                neurons_mem.append(mac_mem)
                with open(fname_mac_result,'a', newline='', encoding='utf-8') as file:
                    file_wr = csv.writer(file, delimiter=",")
                    file_wr.writerow([datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S"),
                                      sample,
                                      neuron,
                                      wl_plus,
                                      wl_minus,
                                      xp_gp,
                                      xm_gp,
                                      xp_gm,
                                      xm_gm,
                                      scale_x,
                                      mac_mem,
                                      inputs @ self.layer_weights[:, neuron]])
            all_neurons_mem.append(neurons_mem)
        self.device.set_wr()
        all_neurons_mem = np.array(all_neurons_mem)
        return all_neurons_mem[0] if single else all_neurons_mem

    def _dot_product(self, x, wl) -> float:
        """
        Dot product of positive voltages vector with crossbar column, skip zero vectors
        """
        if not np.any(x):
            return 0
        return self.device.dot_product(x, wl)[0]

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
