"""
Ядро обработки
"""

import os
import pickle
import numpy as np

from MemriNeurons.src import get_logger, change_log_file
from manager.app import Application
from manager.service import a2r, r2w, v2d, a2v

# pylint: disable=C0301

class HardCore(Application):
    """
    Функции вычислителя
    """

    COL_NUM = 8
    ROW_NUM = 32
    WRITE_ATTEMPTS = 10
    V_START = 0.3
    V_RESET = 3
    V_SET = 3
    V_STEP = 0.05

    logger = None

    def __init__(self, conn):
        """
        Инициализация вычислителя
        """
        super().__init__()
        self.conn = conn

    def update_logger(self):
        """
        Задать логирование в файл
        """
        if self.logger is None:
            self.logger = get_logger()
        else:
            change_log_file(self.logger)

    def read_one_weight(self, wl, bl):
        """
        Прочитать одну ячейку
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
        return r2w(self.res_load, res)

    def read_raw_weights(self):
        """
        Прочитать все веса в кроссбаре
        """
        weights = np.zeros((self.ROW_NUM, self.COL_NUM))
        for bl in range(self.ROW_NUM):
            for wl in range(self.COL_NUM):
                weights[bl][wl] = self.read_one_weight(wl, bl)
        return weights

    def write_weight(self, wl, bl, weight_value):
        """
        Записать вес
        В симуляторе веса меняются от 0.07 до 0.33
        """
        for _ in range(self.WRITE_ATTEMPTS):
            current_weight = self.read_one_weight(wl, bl)
            if current_weight > weight_value: # уменьшаем
                for vol in np.arange(self.V_START, self.V_RESET+self.V_STEP, self.V_STEP):
                    vol_dac = v2d(self.dac_bit, self.vol_ref_dac, vol)
                    adc = self.conn.mode_7(vol_dac, 0, 125, 0, 0, wl, bl)
                    adc = adc[0]
                    res = a2r(self.gain, self.res_load, self.vol_read, self.adc_bit, self.vol_ref_adc, self.res_load, adc)
                    current_weight = r2w(self.res_load, res)
                    if current_weight < weight_value: # прерываем
                        break
            else: # увеличиваем
                for vol in np.arange(self.V_START, self.V_SET+self.V_STEP, self.V_STEP):
                    vol_dac = v2d(self.dac_bit, self.vol_ref_dac, vol)
                    adc = self.conn.mode_7(vol_dac, 0, 125, 1, 0, wl, bl)
                    adc = adc[0]
                    res = a2r(self.gain, self.res_load, self.vol_read, self.adc_bit, self.vol_ref_adc, self.res_load, adc)
                    current_weight = r2w(self.res_load, res)
                    if current_weight > weight_value: # прерываем
                        break
        return current_weight

    def write_matrix(self, matrix):
        """
        Записать матрицу
        """
        # pylint: disable=C0200
        for bl in range(len(matrix)):
            for wl in range(len(matrix[bl])):
                self.write_weight(wl, bl, matrix[bl][wl])

    def read_mem_weights(self, save_folder='', silent=True, weight_correction=1, dop_mod=False, diap=False, cells_filter=False, **kwargs):
        """
        Прочитать все веса в сети
        """
        all_mem_weights = np.zeros(shape=(self.ROW_NUM, self.COL_NUM), dtype=float)
        for wl in range(self.COL_NUM):
            for bl in range(self.ROW_NUM):
                adc = self.conn.mode_7(0, 0, 0, 0, 0, wl, bl)
                adc = adc[0]
                # повторный запрос
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
                                print(f'wl {wl} bl {bl} = {int(res)}, {r2w(self.res_load, int(res)) * weight_correction}')
                            all_mem_weights[bl][wl] = r2w(self.res_load, int(res)) * weight_correction
                    if cells_filter:
                        if not (wl, bl) in kwargs['cells']:
                            all_mem_weights[bl][wl] = 1000000
                        else:
                            if not silent:
                                print(f'wl {wl} bl {bl} = {int(res)}, {r2w(self.res_load, int(res)) * weight_correction}')
                            all_mem_weights[bl][wl] = r2w(self.res_load, int(res)) * weight_correction
                else:
                    if not silent:
                        print(f'wl {wl} bl {bl} = {int(res)}, {r2w(self.res_load, int(res)) * weight_correction}')
                    all_mem_weights[bl][wl] = r2w(self.res_load, int(res)) * weight_correction
        if save_folder:
            with open(os.path.join(save_folder, 'all_mem_weights.pkl'), 'wb') as fp:
                pickle.dump(all_mem_weights, fp)

        return all_mem_weights

    def multiply(self, x, scale_x, sign_x, scale_w, sign_w, wl, bl):
        """
        Умножение
        """
        input_mem = int(np.round(abs(x)/scale_x*self.vol_read*2**self.dac_bit/self.vol_ref_dac))
        adc = self.conn.mode_9(input_mem, 0, wl, bl)[0]
        mul = a2v(self.gain,
                  self.adc_bit,
                  self.vol_ref_adc,
                  adc)
        mul_res = mul * sign_x * sign_w / scale_w / self.vol_read * scale_x
        return mul_res, adc
