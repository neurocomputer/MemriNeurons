"""
Processing core
"""

import os
import pickle
import copy
import numpy as np

from MemriNeurons.src import get_logger, change_log_file
from manager.app import Application
from manager.service import a2r, r2w, v2d, a2v, d2v

# pylint: disable=C0301

class HardCore(Application):
    """
    Calculator functions
    """

    COL_NUM = 8
    ROW_NUM = 32
    WRITE_ATTEMPTS = 10
    V_START = 0.3 # V
    V_RESET = 3 # V
    V_SET = 3 # V
    V_STEP = 0.01 # V
    T_US = 125 # us
    MAX_SUM_CURRENT = 15 # mA
    MAX_VOL_INFERENCE = 0.3 # V

    logger = None

    def __init__(self, conn):
        """
        Calculator initialization
        """
        super().__init__()
        self.conn = conn
        self.max_dac_value = v2d(self.dac_bit, self.vol_ref_dac, self.MAX_VOL_INFERENCE)
        self.current_weights = np.zeros(shape=(self.ROW_NUM, self.COL_NUM))
        self.current_resistances = np.zeros(shape=(self.ROW_NUM, self.COL_NUM))
        self.set_wr()

    def set_wr(self):
        """
        Switch to write-read mode
        """
        self._contour = "wr"
        self.read_raw_weights()

    def set_mvm(self):
        """
        Switch to matrix-vector multiplication mode
        """
        self._contour = "mvm"
        self.read_raw_weights()

    def update_logger(self):
        """
        Set logging to file
        """
        if self.logger is None:
            self.logger = get_logger()
        else:
            change_log_file(self.logger)

    def _calculate_weight_value(self, res):
        """
        Calculate the weight value
        """
        if self._contour == 'wr':
            return r2w(self.res_load, res)
        elif self._contour == 'mvm':
            return self.sum_gain / res # todo: move to manager.service

    def read_one_weight(self, bl, wl):
        """
        Read a single cell
        """
        res = self.measure_resistance(bl, wl)
        weight = self._calculate_weight_value(res)
        self.current_weights[bl][wl] = weight
        self.current_resistances[bl][wl] = res
        return weight, res

    def read_raw_weights(self):
        """
        Read all weights in the crossbar
        """
        weights = np.zeros((self.ROW_NUM, self.COL_NUM))
        for bl in range(self.ROW_NUM):
            for wl in range(self.COL_NUM):
                weights[bl][wl], _ = self.read_one_weight(bl, wl)
        return weights

    def write_weight(self, bl, wl, weight_value, silent=True):
        """
        Write a weight
        In the simulator, weights range from 0.07 to 0.33
        """
        vol_history = []
        weight_history = []
        need_break = False
        weight_value = abs(weight_value)
        for _ in range(self.WRITE_ATTEMPTS):
            current_weight, _ = self.read_one_weight(bl, wl)
            # weight_history.append(current_weight)
            if current_weight > weight_value: # decrease
                for vol in np.arange(self.V_START, self.V_RESET+self.V_STEP, self.V_STEP):
                    res = self.apply_voltage(vol, 0, bl, wl)
                    current_weight = self._calculate_weight_value(res)
                    current_weight, _ = self.read_one_weight(bl, wl) # todo: investigate!
                    weight_history.append(current_weight)
                    vol_history.append(float(vol))
                    if not silent:
                        print(f'Goal weight {weight_value}, Current weight {current_weight}')
                    if current_weight < weight_value: # break
                        need_break = True
                        break
            else: # increase
                for vol in np.arange(self.V_START, self.V_SET+self.V_STEP, self.V_STEP):
                    res = self.apply_voltage(vol, 1, bl, wl)
                    current_weight = self._calculate_weight_value(res)
                    current_weight, _ = self.read_one_weight(bl, wl) # todo: investigate!
                    weight_history.append(current_weight)
                    vol_history.append(-float(vol))
                    if not silent:
                        print(f'Goal weight {weight_value}, Current weight {current_weight}')
                    if current_weight > weight_value: # break
                        need_break = True
                        break
            if need_break:
                break
        current_weight, _ = self.read_one_weight(bl, wl)
        # weight_history.append(current_weight)
        return current_weight, weight_history, vol_history

    def write_matrix(self, matrix, silent=True):
        """
        Write a matrix
        """
        # pylint: disable=C0200
        assert len(matrix) <= self.ROW_NUM
        assert len(matrix[0]) <= self.COL_NUM
        for bl in range(len(matrix)):
            for wl in range(len(matrix[bl])):
                _ = self.write_weight(bl, wl, matrix[bl][wl])
                if not silent:
                    print(f'Goal weight {matrix[bl][wl]}, Current weight {self.current_weights[bl][wl]}, Current resistance {self.current_resistances[bl][wl]}')
        return copy.deepcopy(self.current_weights), copy.deepcopy(self.current_resistances)

    def read_mem_weights(self, save_folder='', silent=True, weight_correction=1, dop_mod=False, diap=False, cells_filter=False, **kwargs):
        """
        Read all weights in the network
        """
        all_mem_weights = np.zeros(shape=(self.ROW_NUM, self.COL_NUM), dtype=float)
        for wl in range(self.COL_NUM):
            for bl in range(self.ROW_NUM):
                adc = self.conn.mode_7(0, 0, 0, 0, 0, wl, bl)
                adc = adc[0]
                # retry request
                if adc < 10:
                    adc = self.conn.mode_7(0, 0, 0, 0, 0, wl, bl)
                    adc = adc[0]
                adc_value = adc
                if adc_value < 50:
                    adc_value = 50
                res = a2r(self.gain,
                          self.res_load,
                          self.vol_read,
                          self.adc_bit,
                          self.vol_ref_adc,
                          self.res_switches,
                          adc)
                if res <= 0:
                    res = 0.00000001
                if dop_mod:
                    if diap:
                        if kwargs['diap_min'] > res or res > kwargs['diap_max']:
                            all_mem_weights[bl][wl] = 1000000
                        else:
                            if not silent:
                                print(f'bl {bl} wl {wl} = {int(res)}, {self._calculate_weight_value(res) * weight_correction}')
                            all_mem_weights[bl][wl] = self._calculate_weight_value(res) * weight_correction
                    if cells_filter:
                        if not (bl, wl) in kwargs['cells']:
                            all_mem_weights[bl][wl] = 1000000
                        else:
                            if not silent:
                                print(f'bl {bl} wl {wl} = {int(res)}, {self._calculate_weight_value(res) * weight_correction}')
                            all_mem_weights[bl][wl] = self._calculate_weight_value(res) * weight_correction
                else:
                    if not silent:
                        print(f'bl {bl} wl {wl} = {int(res)}, {self._calculate_weight_value(res) * weight_correction}')
                    all_mem_weights[bl][wl] = self._calculate_weight_value(res) * weight_correction
        if save_folder:
            with open(os.path.join(save_folder, 'all_mem_weights.pkl'), 'wb') as fp:
                pickle.dump(all_mem_weights, fp)

        return all_mem_weights

    def multiplication(self, x, bl, wl, scale_x=1, scale_w=1, sign_w=1):
        """
        Multiplication. Performed via read/write loop
        x - voltage
        """
        if self._contour == 'wr':
            sign_x = np.sign(x)
            input_mem = v2d(self.dac_bit, self.vol_ref_dac, abs(x)/scale_x)
            if input_mem > self.max_dac_value:
                input_mem = self.max_dac_value
                print("ATTENTION! You tried to apply grater 0.3 V to a memristor during inference!")
            adc = self.conn.mode_9(input_mem, 0, wl, bl)[0]
            mul = a2v(self.gain,
                    self.adc_bit,
                    self.vol_ref_adc,
                    adc)
            mul_res = mul * sign_x * scale_x * sign_w * scale_w
            return mul_res, adc
        else:
            print(f"ATTENTION! You tried to work in wr-mode, but {self._contour}-mode set up!")

    def dot_product(self, x, wl, scale_x=1, scale_w=1):
        """
        Dot product of two vectors
        """
        if self._contour == 'mvm':
            assert len(x) <= self.ROW_NUM
            v_dac = [0 for i in range(self.ROW_NUM)]
            # safety check
            for h, x_value in enumerate(x):
                input_mem = v2d(self.dac_bit, self.vol_ref_dac, abs(x_value)/scale_x)
                if input_mem > self.max_dac_value:
                    v_dac[h] = self.max_dac_value
                    print("ATTENTION! You tried to apply grater 0.3 V to a memristor during inference!")
                else:
                    v_dac[h] = input_mem
            # predict for safety
            resistances_vector = self.current_resistances[:,wl]
            voltage_vector = list(map(lambda x: d2v(self.dac_bit, self.vol_ref_dac, x), v_dac))
            predicted_current = np.sum(voltage_vector/resistances_vector)
            if predicted_current*1000 >= self.MAX_SUM_CURRENT:
                v_dac = [0 for i in range(len(x))]
                print(f"ATTENTION! Sims the result of dot product will be grater than {self.MAX_SUM_CURRENT}mA!")
            # self, vDAC_mas, tms, tus, rtms, rums, wl, id):
            adc = self.conn.mode_mvm(v_dac, 0, 0, 0, 0, wl, 0)[0]
            mul = a2v(1,
                    self.adc_bit,
                    self.vol_ref_adc,
                    adc)
            mul_res = mul * scale_w
            return mul_res, adc
        else:
            print(f"ATTENTION! You tried to work in mvm-mode, but {self._contour}-mode set up!")

    def apply_voltage(self, vol, rev, bl, wl):
        """
        Подать напряжение
        """
        vol_dac = v2d(self.dac_bit, self.vol_ref_dac, vol)
        adc = self.conn.mode_7(vol_dac, 0, self.T_US, rev, 0, wl, bl)
        adc = adc[0]
        res = a2r(self.gain, self.res_load, self.vol_read, self.adc_bit, self.vol_ref_adc, self.res_load, adc)
        return res

    def measure_resistance(self, bl, wl):
        """
        Измерить сопротивление
        """
        adc = self.conn.mode_7(0, 0, 0, 0, 0, wl, bl)
        adc = adc[0]
        res = a2r(self.gain,
                  self.res_load,
                  self.vol_read,
                  self.adc_bit,
                  self.vol_ref_adc,
                  self.res_switches,
                  adc)
        return res
