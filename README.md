# MemriNeurons

Пока работает только из под MemriBoard, то есть нужно клонировать MemriBoard, в него клонировать MemriNeurons и создавать скрипты на уровне MemriBoard.

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

source_model_path = 'my_model.keras'
new_model_path = 'my_model.custom'

new_model = convert_keras_2_nmp(source_model_path, new_model_path) # конвертация модели
new_model = load_model(new_model_path) # загрузка модели

conn = BoardSimulator() # подключаем плату
_ = conn.connect('simulator')

device = HardCore(conn) # создаем ядро

hardlayer = ElementWiseMatMulLayer(device, 'Dense_0', save_folder='my_model') # создаем слой
hardlayer.find_weights_model(new_model.layers[0].get_weights(), 0.33)

new_model.layers[0].matmul = hardlayer.matmul # подменяем функцию матричного умножения
output = new_model.predict([x])
