"""
Спайковая сеть
"""

import numpy as np
from MemriNeurons.src import get_logger, change_log_file

class Neuron():

    logger = None

    def __init__(self, device, synapses, tresh, leakage):
        self.synapses = synapses
        self.tresh = tresh
        self.device = device
        self.leakage = leakage

    def update_logger(self):
        """
        Задать логирование в файл
        """
        if self.logger is None:
            self.logger = get_logger()
        else:
            change_log_file(self.logger)

    def run(self, input_data, stdp=False):
        output = []
        membrain = 0
        membrain_values = []
        weights_history = np.zeros((len(self.synapses), len(input_data[0])))
        for i in range(len(input_data[0])):
            for j, signal in enumerate(input_data):
                bl = self.synapses[j][0]
                wl = self.synapses[j][1]
                if signal[i] != 0:
                    scale_x = 1
                    sign_x = 1
                    scale_w = 1
                    sign_w = 1
                    mul_res, adc = self.device.multiply(signal[i],
                                                       scale_x,
                                                       sign_x,
                                                       scale_w,
                                                       sign_w,
                                                       wl,
                                                       bl)
                    membrain+=mul_res
            if membrain > self.tresh:
                output.append(1)
                membrain_values.append(membrain)
                membrain = 0
                # if stdp:
                #     for j, signal in enumerate(input_data):
                #         bl = self.synapses[j][0]
                #         wl = self.synapses[j][1]
                #         if signal[i] != 0: # усиливаем связь
                #             # adc = mode_7(SERIAL, self.conn, wl=wl, bl=bl, vol_read=0.3, vol=-V_STDP, res_load=3000, res_switches=10, gain=11, adc_bit=14, vol_ref_adc=5, duration=T_STPD)
                #             adc, _ = self.conn.mode_7(V_STDP, 0, 125, 1, 0, wl, bl)
                #         else: # ослабляем связь
                #             #adc = mode_7(SERIAL, self.conn, wl=wl, bl=bl, vol_read=0.3, vol=V_STDP, res_load=3000, res_switches=10, gain=11, adc_bit=14, vol_ref_adc=5, duration=T_STPD)
                #             adc, _ = self.conn.mode_7(V_STDP, 0, 125, 0, 0, wl, bl)
                #         res = convert_adc_to_res(11, 3000, 0.3, 14, 5, 10, adc)
                #         current_weight = convert_res_to_weight(3000, res)
                #         weights_history[j][i] = current_weight
            else:
                # if stdp:
                #     for j, signal in enumerate(input_data):
                #         bl = self.synapses[j][0]
                #         wl = self.synapses[j][1]
                #         weights_history[j][i] = read_one_weight(self.conn, wl, bl)
                output.append(0)
                membrain -= self.leakage
                if membrain < 0: membrain = 0
                membrain_values.append(membrain)

        return membrain_values, output, weights_history
