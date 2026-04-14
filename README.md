# MemriNeurons

Для запуска необходимо сначала склонировать репозиторий MemriBoard, затем в него склонировать репозиторий MemriNeurons и если вы работаете с настоящей платой, а не симулятором, то нужно еще склонировать репозиторий MemriCore. Нейросети создавать в папке MemriBoard:

MemriBoard
        |-MemriNeurons
        |-MemriCORE
        |-simulator.cb
        |-ваш_файл.py

Пример работы в файле xor_example.py. Для его запуска его нужно скопировать на уровень ваш_файл.py. Внутри файла нужно установить в 1 те тесты, которые хотите выполнить.

Структура MemriNeurons:

components.py - содержит компоненты ИНС для конвертации, чтобы можно было запускать модель только с помощю numpy:
-- load_model - загрузка модели
-- feature_map_reshape - преобразователь карт признаков для обработки в сверточном слое
-- Sequential - модель ИНС
-- Layer - слой ИНС
-- Dense - полносвязный слой
-- Conv2D - сверточный слой
-- Flatten - слой выпрямления

keras2nmp.py - конвертер модели ИНС из формата keras в формат numpy
-- convert_keras_2_nmp - функция конвертации

hardlayers.py - содержит классы, выполняющие матричное умножение в слоях ИНС:
-- ElementWiseMatMulLayer - поэлементное умножение для целого слоя

cores.py - вычислитель на кроссбар массиве мемристоров. Есть плата, которая содержит мемристоры. Для взаимодействия с платой используется драйвер из MemriCore, но он выполняет лишь пересылку данных по интерфейсу (в режимах mode_7, mode_9, mode_mvm, mode_core). Но ядру нужны и дополнительные функции, такие как чтение/запись весов и т.д.:
-- HardCore - класс работы с ядром

src.py - содержит вспомогательные функции
-- get_logger - получить логгер
-- change_log_file - сменить лог файл (использует get_file_handler)
-- poisson_binary_array - создает массив нулей и единиц по распределению Пуассона

## Пример создания ИНС

from simulator.src import BoardSimulator
from MemriNeurons.keras2nmp import convert_keras_2_nmp
from MemriNeurons.components import load_model
from MemriNeurons.cores import HardCore
from MemriNeurons.hardlayers import ElementWiseMatMulLayer

SOURCE_MODEL_PATH = 'my_model.keras'
NEW_MODEL_PATH = 'my_model.custom'

new_model = convert_keras_2_nmp(SOURCE_MODEL_PATH, NEW_MODEL_PATH) # конвертация модели
new_model = load_model(NEW_MODEL_PATH) # загрузка модели

conn = BoardSimulator() # подключаем плату
_ = conn.connect('simulator') # должен быть создан simulator.cb

device = HardCore(conn) # создаем ядро

hardlayer1 = ElementWiseMatMulLayer(device, 'Dense_1', save_folder='my_model') # создаем слой
hardlayer1.find_weights_model(new_model.layers[0].get_weights(), 0.33)
new_model.layers[0].matmul = hardlayer1.matmul # подменяем функцию матричного умножения

hardlayer2 = ElementWiseMatMulLayer(device, 'Dense_2', save_folder='my_model') # создаем слой
hardlayer2.find_weights_model(new_model.layers[1].get_weights(), 0.33)
new_model.layers[1].matmul = hardlayer2.matmul # подменяем функцию матричного умножения

output = new_model.predict(input_data)
