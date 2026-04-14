import os
import datetime
import logging

def change_log_file(logger):
    """
    Изменяет файл для записи логов
    """
    if logger.handlers:
        logger.handlers.clear()
    file_handler = get_file_handler()
    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)

def get_file_handler():
    """
    Получить для логгера файл хэндлер
    """
    delimiter = ','
    fmt = f'%(asctime)s{delimiter}%(message)s'
    filename = os.path.join('logs', datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    formatter = logging.Formatter(
        fmt,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Создаем файловый обработчик
    file_handler = logging.FileHandler(filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    return file_handler

def get_logger():
    """
    Добавить логгер
    """
    logger = logging.getLogger()
    level = logging.INFO
    logger.setLevel(level)
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = get_file_handler()
    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)
    return logger

def poisson_binary_array(size, lambda_rate):
    """
    Создает массив из нулей и единиц, где единицы распределены по закону Пуассона.
    """
    poisson_arr = np.random.poisson(lam=lambda_rate, size=size)
    binary_arr = (poisson_arr > 0).astype(int)
    
    return binary_arr
