import os
import shutil
import json # Добавлено для возможности загрузки конфига при необходимости
import torch # <<< ДОБАВЛЕНО: Импорт torch
import numpy as np # Добавлено для plot_spectrogram и, возможно, других нужд
import matplotlib # Импорт matplotlib
# Убедитесь, что 'Agg' backend установлен, если вы используете его без GUI
# matplotlib.use("Agg") # Эту строку можно убрать или оставить, предупреждение может все равно появляться если backend уже выбран
import matplotlib.pyplot as plt # Импорт pyplot для plot_spectrogram


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# Измененная функция build_env
# Теперь она просто создает указанную директорию, если она не существует как директория.
# Убрана логика копирования конфига, так как она, вероятно, должна быть в другом месте.
def build_env(save_directory):
    """
    Создает директорию для сохранения моделей и логов, если она не существует.

    Args:
        save_directory (str): Путь к директории, которую нужно создать.
    """
    # Исправление: Проверяем, существует ли путь И является ли он директорией.
    if not os.path.exists(save_directory) or not os.path.isdir(save_directory):
        try:
            # recursive=True deprecated in Python 3.10+, use parents=True
            # exist_ok=True предотвращает ошибку, если директория уже существует
            os.makedirs(save_directory, exist_ok=True)
            print(f"Создана директория для сохранения: {save_directory}")
        except OSError as e:
            print(f"Ошибка при создании директории {save_directory}: {e}")
            raise # Перевызываем исключение после печати ошибки
    else:
         # Это информационное сообщение, а не ошибка
         print(f"Директория для сохранения уже существует: {save_directory}")


# Добавим функции load_checkpoint, save_checkpoint, scan_checkpoint и plot_spectrogram сюда,
# чтобы их можно было импортировать в train.py без конфликтов и чтобы убрать их локальные определения.
# Это сделает код более модульным. Скопируем их из вашего train.py.

def scan_checkpoint(cp_dir, prefix):
    """
    Сканирует директорию на наличие файлов чекпоинтов с заданным префиксом
    и возвращает путь к самому последнему (по имени) файлу.
    Если cp_dir является файлом и соответствует префиксу, возвращает его.
    """
    if os.path.isdir(cp_dir):
        # Используем listdir и join для формирования полных путей
        checkpoints = [os.path.join(cp_dir, f) for f in os.listdir(cp_dir) if f.startswith(prefix)]
        if not checkpoints:
            return None
        # Сортируем по имени для нахождения последнего (предполагается, что имена содержат номер шага)
        return sorted(checkpoints)[-1]

    # Если переданный путь - это файл
    if os.path.isfile(cp_dir):
        basename = os.path.basename(cp_dir)
        if basename.startswith(prefix):
            return cp_dir # Возвращаем сам путь к файлу

    return None


def load_checkpoint(filepath, device):
    """
    Загружает чекпоинт из файла.
    """
    # Добавлена проверка существования файла
    if not os.path.isfile(filepath):
        print(f"Предупреждение: Файл чекпоинта не найден: {filepath}")
        return None # Возвращаем None, если файл не найден

    print("Loading checkpoint '{}'".format(filepath))
    try:
        # Используется torch.load, теперь torch импортирован
        checkpoint_dict = torch.load(filepath, map_location=device)
        return checkpoint_dict
    except Exception as e:
        print(f"Ошибка при загрузке чекпоинта {filepath}: {e}")
        return None # Возвращаем None при ошибке загрузки


def save_checkpoint(filepath, obj):
    """
    Сохраняет объект в файл чекпоинта.
    """
    # Добавлена проверка, существует ли родительская директория
    parent_dir = os.path.dirname(filepath)
    # Если parent_dir пустой (например, если filepath - просто имя файла в текущей директории),
    # не пытаемся создать пустую директорию
    if parent_dir and not os.path.exists(parent_dir):
        print(f"Родительская директория для сохранения чекпоинта не существует: {parent_dir}. Попытка создать...")
        # Попытка создать директорию, если ее нет
        try:
            os.makedirs(parent_dir, exist_ok=True)
            print(f"Создана родительская директория для чекпоинта: {parent_dir}")
        except OSError as e:
             print(f"Не удалось создать родительскую директорию {parent_dir}: {e}")
             # Можно решить, что делать дальше: вызвать ошибку или просто не сохранять
             print(f"Предупреждение: Сохранение чекпоинта {filepath} пропущено из-за ошибки создания директории.")
             return # Прекращаем сохранение, если директория не создана


    print("Saving checkpoint to {}".format(filepath))
    try:
        torch.save(obj, filepath) # Используется torch.save, теперь torch импортирован
    except Exception as e:
        print(f"Ошибка при сохранении чекпоинта {filepath}: {e}")


def plot_spectrogram(spectrogram):
    """Plots a spectrogram (numpy array)."""
    # Добавлена проверка, является ли входной массив numpy
    if not isinstance(spectrogram, np.ndarray):
        print("Ошибка plot_spectrogram: Ожидается numpy.ndarray")
        return None # Возвращаем None при некорректном входе

    # Проверка на пустой массив
    if spectrogram.size == 0:
         print("Предупреждение plot_spectrogram: Входной массив пуст.")
         return None


    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    return fig
