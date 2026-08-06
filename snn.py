"""
Спайковая сеть

Рефракторный период - переключаемся из контура МВУ в контур записи и делаем обновление
"""

import numpy as np
from MemriNeurons.src import get_logger, change_log_file

class Astrocyte():

    membrain: int = 0 # мембрана (количество спайков)
    tresh: float = 100 # трешхолд (количество спайков)
    active_time: int = 40 # активен столько тактов
    active: int = 0 # активность астроцита
    membrains_changers: list # изменяторы трешхолда
    gain = 0.3 # коэффициент изменения трешхолда
    membrain_history: list # история астроцита

    def __init__(self):
        self.membrains_changers = []
        self.membrain_history = []

    def set_controlled_membrains(self, neurons):
        """
        Установить контроллируемые мембраны
        """
        for neuron in neurons:
            self.membrains_changers.append(neuron.change_membrain_tresh)

    def set_input_spikes(self, values):
        """
        Подать на вход
        """
        if self.active == 0: # не активен
            for value in values: # накапливает потенциал
                if value > 0:
                    self.membrain += value
            if self.membrain > self.tresh:
                self.active = self.active_time
                self.activate_astrocyte() # активируем
                self.membrain_history.append(1) # активен
            else:
                self.membrain_history.append(0) # не активен
        else:
            self.active -= 1
            if self.active == 0: # деактивируем
                self.membrain_history.append(0)
                self.deactivate_astrocyte()
            else:
                self.membrain_history.append(1) # активен

    def activate_astrocyte(self):
        """
        Обновить активность астроцита
        """
        for membrain_tresh_changer in self.membrains_changers:
            membrain_tresh_changer(mode='gain', gain=self.gain)

    def deactivate_astrocyte(self):
        """
        Деактивируем астроцит
        """
        self.membrain = 0
        for membrain_tresh_changer in self.membrains_changers:
            membrain_tresh_changer(mode='basic')

class Synapse():
    """
    Synapse
    """

    logger = None

    trace_value: int = 0 # начальное значение трейса
    trace_deep: int = 4 # ширина трейса (в импульсах)
    trace_history: list # история трейса
    conductace_history: list # история проводимости мемристора
    weight_history: list # история изменения веса
    lock_stdp: bool = False # блокер STDP

    def __init__(self, device, logger, bl, wl, neuron_name):
        self.device = device # устройство подключения
        self.logger = logger # логгер
        self.bl = bl
        self.wl = wl
        self.neuron_name = neuron_name
        self.trace_history = [] # история трейса
        self.conductace_history = [] # история проводимости
        self.weight_history = [] # история веса

    def update_trace(self, spike):
        """
        Обновить трейс
        """
        if spike > 0: # если спайк больше 0 то увеличиваем трейс
            self.trace_value = self.trace_deep
        else:
            self.trace_value -= 1
        if self.trace_value < 0:
            self.trace_value = 0
        self.trace_history.append(self.trace_value)

    def update_conductance(self):
        """
        Обновить проводимость
        """
        self.device.set_mvm()
        weight, res = self.device.read_one_weight(self.bl, self.wl)
        self.conductace_history.append(1/res)
        self.weight_history.append(weight)
        self.device.set_wr()

    def potentiate(self, voltage):
        """
        Потенциация (SET)
        """
        if not self.lock_stdp:
            print(f"{self.neuron_name} синапс {self.bl} - потенциация")
            res = self.device.apply_voltage(voltage, 1, self.bl, self.wl)
        else:
            print(f"{self.neuron_name} синапс {self.bl} - потенциация БЛОКИРОВАНО!")
            res = self.device.measure_resistance(self.bl, self.wl)
        return 1/res

    def depress(self, voltage):
        """
        Депрессия (RESET)
        """
        if not self.lock_stdp:
            print(f"{self.neuron_name} синапс {self.bl} - депрессия")
            res = self.device.apply_voltage(voltage, 0, self.bl, self.wl)
        else:
            print(f"{self.neuron_name} синапс {self.bl} - депрессия БЛОКИРОВАНО!")
            res = self.device.measure_resistance(self.bl, self.wl)
        return 1/res

class LIFNeuron():
    """
    LIF нейрон
    """

    logger = None

    membrain: float = 0.0 # мВ
    membrain_relaxed = 0.0 # значение мембраны по умолчанию (потенциал покоя)
    tresh: float # трешхолд (текущий)
    leakage: float = 0.01/178*19 # утечка
    trace_value: int = 0 # начальное значение трейса
    trace_deep: int = 4 # ширина трейса (в импульсах)
    trace_history: list # история трейса
    membrain_history: list
    output_history: list
    basic_tresh: float = 0.68/178*19 # трешхолд (по умолчанию)

    vol_max_pot = 2.85
    vol_min_pot = 2.4
    vol_max_dep = 2.5
    vol_min_dep = 1.9
    potentiation_table = np.linspace(vol_min_pot, vol_max_pot, trace_deep)
    depression_table = np.linspace(vol_min_dep, vol_max_dep, trace_deep)

    def __init__(self, device, wl, synapses_amount, name='Neuron'):
        assert synapses_amount < device.ROW_NUM
        assert wl < device.COL_NUM
        self.name = name
        self.device = device # устройство исполнения
        self.wl = wl # столбец кроссбара нейрона
        self.update_logger() # получаем логгер
        self.synapses = [Synapse(self.device, self.logger, i, self.wl, self.name) for i in range(synapses_amount)] # синапсы
        self.membrain_history = [] # история мембраны
        self.output_history = [] # история выхода нейрона
        self.trace_history = [] # история трейса
        self.change_membrain_tresh(mode='basic')

    def update_logger(self):
        """
        Задать логирование в файл
        """
        if self.logger is None:
            self.logger = get_logger()
        else:
            change_log_file(self.logger)

    def update_synapse_traces(self, spikes):
        """
        Обновляем трейсы синапсов
        """
        for spike_indx, spike in enumerate(spikes):
            self.synapses[spike_indx].update_trace(spike)

    def update_synapce_conductances(self):
        """
        Записать проводимости синапсов
        """
        self.device.set_wr()
        for synapse in self.synapses:
            synapse.update_conductance() 

    def potentiation(self):
        """
        Потенциация
        """
        self.device.set_wr()
        for synapse in self.synapses:
            if max(0, synapse.trace_value):
                synapse.potentiate(self.potentiation_table[synapse.trace_value-1])

    def depression(self, spikes):
        """
        Депрессия
        """
        self.device.set_wr()
        for spike_indx, spike in enumerate(spikes):
            if spike > 0:
                self.synapses[spike_indx].depress(self.depression_table[self.trace_value-1])

    def set_input_spike(self, spikes, stdp=True):
        """
        Подать один набор спайков
        """
        spikes = spikes * 0.3 # превращаем в 300 мВ
        assert len(spikes) == len(self.synapses) # спайков столько сколько синапсов
        self.update_synapse_traces(spikes) # обновляем трейсы синапсов
        if self.trace_value <= 0: # не рефракторный период
            self.device.set_mvm()
            mul_res = self.device.dot_product(spikes, self.wl)[0]
            self.membrain += mul_res
            self.membrain -= self.leakage
            if self.membrain < self.membrain_relaxed:
                self.membrain = self.membrain_relaxed
            self.membrain_history.append(self.membrain)
            if self.membrain > self.tresh:
                self.membrain = self.membrain_relaxed
                self.trace_value = self.trace_deep + 1
                output_spike = 1
                if stdp: # STDP
                    # потенциация
                    self.potentiation()
            else:
                output_spike = 0
        else: # рефракторный период
            self.membrain_history.append(self.membrain)
            output_spike = 0
            if stdp: # STDP
                # депрессия
                self.depression(spikes)
        self.trace_value -= 1
        if self.trace_value < 0:
            self.trace_value = 0
        self.output_history.append(output_spike)
        self.trace_history.append(self.trace_value)
        self.update_synapce_conductances()
        return output_spike

    def change_membrain_tresh(self, mode, **kwargs):
        """
        Изменить трешхолд в gain раз
        Сбросить значение трешхолда по умолчанию
        """
        if mode == 'gain':
            self.basic_tresh = self.tresh
            self.tresh = self.tresh * kwargs['gain']
            print(f'{self.name} Трешхолд gain {self.tresh}')
        elif mode == 'basic':
            self.tresh = self.basic_tresh
            print(f'{self.name} Трешхолд basic {self.tresh}')
