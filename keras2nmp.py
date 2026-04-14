"""
Преобразователь модели keras в формат без tensorflow
tensorflow используется только здесь
"""

# pylint: disable=E1101

import pickle
import tensorflow as tf
from MemriNeurons.components import Sequential, Dense, Conv2D, Flatten

def convert_keras_2_nmp(source_model_path, target_model_path):
    """
    Преобразователь модели keras в формат без tensorflow
    Пока работает только с 3мя видами слоев Dense Conv2D Flatten
    Метрики не извлекает, только модель
    """
    source_model = tf.keras.models.load_model(source_model_path)
    new_model = Sequential()
    for _, layer in enumerate(source_model.layers):
        # параметры слоя
        layer_name = layer.__class__.__name__
        # запуск обработки слоя
        if layer_name == 'Dense':
            activation_name = layer.activation.__name__
            print('Dense layer found')
            new_model.add(Dense(weights = layer.get_weights(),
                                activation = activation_name
                                ))
        elif layer_name == 'Conv2D':
            activation_name = layer.activation.__name__
            print('Conv2D layer found')
            new_model.add(Conv2D(weights = layer.get_weights(),
                                 activation = activation_name,
                                 strides = layer.strides))
        elif layer_name == 'Flatten':
            print('Flatten layer found')
            new_model.add(Flatten())
        else:
            print(f'Не известный слой {layer_name}!')
            return None
    # сохранение модели
    with open(target_model_path, 'wb') as handle:
        pickle.dump(new_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return new_model
