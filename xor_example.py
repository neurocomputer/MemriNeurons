"""
Тест запуска нейронной сети на мемристорах
Нейронная сеть: XOR
"""

import os
import numpy as np

from simulator.src import BoardSimulator # коннект к плате
from MemriNeurons.cores import HardCore # ядро
from MemriNeurons.hardlayers import ElementWiseMatMulLayer # слой
from MemriNeurons.components import Activations # функции активации
from MemriNeurons.keras2nmp import convert_keras_2_nmp # конвертер
from MemriNeurons.components import load_model # загрузчик модели

# настройки отображения табличек
np.set_printoptions(precision=4, suppress=True, formatter={'all': lambda x: f'{x:0.4f}'})

# выберете какой тест провести
TEST_READ_WRITE = 0 # проверка записи и чтения весов
TEST_MULTIPLY = 0 # проверка умножения
TEST_ANN = 0 # тест ИНС собранной вручную
TEST_ANN_TF = 0 # тест ИНС с автоматической конвертацией

# подключаем плату
CONN = BoardSimulator()
_ = CONN.connect('simulator')

# создаем обработчик ядра
device = HardCore(CONN)

if TEST_READ_WRITE: # тесты чтения и записи
    WEIGHT = 0.2
    WL = 0
    BL = 0
    print(f'\nВес {BL}, {WL} = {device.read_one_weight(BL, WL)}')
    print(f'Пишем вес {WEIGHT}')
    device.write_weight(BL, WL, WEIGHT)
    print(f'Вес {BL}, {WL} = {device.read_one_weight(BL, WL)}')
    print('\nВсе веса:')
    print(device.read_raw_weights())
    print('\nЗапишем рандомные веса:')
    random_weights = np.random.uniform(0.07, 0.33, size=(32, 8))
    print(random_weights)
    device.write_matrix(random_weights)
    print('\nЗаписанные веса:')
    print(device.read_raw_weights())
    WEIGHT_CORRECTION = 10
    print(f'\nЧтение весов с коэффициентом {WEIGHT_CORRECTION}:')
    if not os.path.exists('xor'):
        os.mkdir('xor')
    _ = device.read_mem_weights(save_folder='xor',
                                silent=False,
                                weight_correction=WEIGHT_CORRECTION)

if TEST_MULTIPLY: # тест уможения
    BL = 0
    WL = 0
    print(f'\nВес {BL}, {WL} = {device.read_one_weight(BL, WL)}')
    X = 1
    print(f'Умножение на X={X}')
    print(f'Результат: {device.multiply(X, 1, 1, 1, 1, BL, WL)}')

if TEST_ANN: # двухслойная сеть собранная вручную
    if not os.path.exists('xor'):
        os.mkdir('xor')

    w1 = np.array([[3.6725047, 5.21765],
                   [3.6788058, 5.24961]], dtype=float)
    b1 = np.array([-5.5291667, -2.2139354], dtype=float)
    w2 = np.array([[-5.7410617],
                   [5.4596796]], dtype=float)
    b2 = np.array([-2.4699624], dtype=float)

    sigmoid = Activations().sigmoid

    # обработчик слоя
    hardlayer1 = ElementWiseMatMulLayer(device, 'Dense_1', save_folder='xor')
    hardlayer1.find_weights_model([w1, b1], 0.33)

    hardlayer2 = ElementWiseMatMulLayer(device, 'Dense_2', save_folder='xor')
    hardlayer2.find_weights_model([w2, b2], 0.33)

    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    out1_etalon = sigmoid(x @ w1 + b1)
    out1 = sigmoid(hardlayer1.matmul(x))
    print('Layer 1')
    for i, item in enumerate(out1):
        print(item, out1_etalon[i])
    out2_etalon = sigmoid(out1_etalon @ w2 + b2)
    out2 = sigmoid(hardlayer2.matmul(out1))
    print('Layer 2')
    for i, item in enumerate(out2):
        print(item, out2_etalon[i])

if TEST_ANN_TF: # загрузка сети из файла

    SOURCE_MODEL_PATH = os.path.join('MemriNeurons', 'XOR.keras')
    NEW_MODEL_PATH = 'XOR.custom'

    new_model = convert_keras_2_nmp(SOURCE_MODEL_PATH, NEW_MODEL_PATH) # конвертация модели
    new_model = load_model(NEW_MODEL_PATH) # загрузка модели

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    output_etalon = new_model.predict(x) # эталонный выход

    conn = BoardSimulator() # подключаем плату
    _ = conn.connect('simulator') # должен быть создан simulator.cb

    device = HardCore(conn) # создаем ядро

    hardlayer1 = ElementWiseMatMulLayer(device, 'Dense_1', save_folder='xor') # создаем слой
    hardlayer1.find_weights_model(new_model.layers[0].get_weights(), 0.33) # маппим веса
    new_model.layers[0].matmul = hardlayer1.matmul # подменяем функцию матричного умножения

    hardlayer2 = ElementWiseMatMulLayer(device, 'Dense_2', save_folder='xor') # создаем слой
    hardlayer2.find_weights_model(new_model.layers[1].get_weights(), 0.33) # маппим веса
    new_model.layers[1].matmul = hardlayer2.matmul # подменяем функцию матричного умножения

    output = new_model.predict(x)
    for i, item in enumerate(output):
        print(item, output_etalon[i])
