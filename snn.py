"""
Спайковая сеть
"""

import numpy as np
from MemriNeurons.src import get_logger, change_log_file

class LIFNeuron():
    """
    LIF нейрон
    """

    logger = None
    scale_x = 1
    sign_x = 1
    scale_w = 1
    sign_w = 1
    membrain: float
    membrain_history: list
    time_distance = 10 # сколько спайков храним в истории
    past_distances: list
    future_distance: list
    output_history: list

    def __init__(self, device, synapses, tresh, leakage):
        self.device = device # устройство исполнения
        self.synapses = synapses # синапсы нейронов [(bl, wl)]
        self.tresh = tresh # трешхолд
        self.leakage = leakage # значение утечки для линейной
        self.membrain = 0.
        self.membrain_history = []
        self.past_distances = [0 for i in range(len(synapses))] # история в прошлое
        self.future_distance = [0 for i in range(len(synapses))] # история в будующее
        self.past_distances_all = [[] for i in range(len(synapses))] # история в прошлое
        self.future_distance_all = [[] for i in range(len(synapses))] # история в будующее
        self.output_history = []

    def update_logger(self):
        """
        Задать логирование в файл
        """
        if self.logger is None:
            self.logger = get_logger()
        else:
            change_log_file(self.logger)

    def set_input_spike(self, spikes, stdp=False):
        """
        Подать один спайк
        """
        assert len(spikes) == len(self.synapses) # спайков столько сколько синапсов
        for synapse_index, spike in enumerate(spikes):
            bl = self.synapses[synapse_index][0]
            wl = self.synapses[synapse_index][1]
            mul_res = 0
            if spike != 0:
                # multiplication(self, x, bl, wl, scale_x=1, scale_w=1, sign_w=1)
                mul_res, _ = self.device.multiplication(spike,
                                                        bl,
                                                        wl,
                                                        self.scale_x,
                                                        self.scale_w,
                                                        self.sign_w,
                                                        )
                self.past_distances[synapse_index] = self.time_distance #0
            else:
                self.past_distances[synapse_index] -= 1
            self.past_distances_all[synapse_index].append(self.past_distances[synapse_index])
            self.membrain += mul_res
        self.membrain_history.append(self.membrain)
        # print(spike, round(mul_res, 3), round(self.membrain,3), self.past_distances)
        if self.membrain > self.tresh:
            self.membrain = 0
            self.output_history.append(1)
            # STDP
            if stdp:
                for synapse_index, synapse in enumerate(self.synapses):
                    bl = synapse[0]
                    wl = synapse[1]
                    print(f'Усиливаем связь на {self.past_distances[synapse_index]}')
        else:
            self.output_history.append(0)
