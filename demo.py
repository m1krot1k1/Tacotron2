# -*- coding: utf-8 -*-
import matplotlib
# Указываем бэкенд, который не требует GUI, ДО импорта pyplot/pylab
matplotlib.use('Agg')
import matplotlib.pylab as plt

import sys
import numpy as np
import torch
import math
import json
import os
import soundfile as sf
from PIL import Image
import time
import streamlit as st

from torch.nn import functional as F

# --- Определение базовой директории ---
# Это важно для корректной работы с относительными путями
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- Попытка импорта зависимостей Tacotron2 ---
# Обернуто в try-except для диагностики проблем с окружением/путями
try:
    from hparams import create_hparams
    from model import Tacotron2
    from train import load_model
    from text import text_to_sequence, symbol_to_id
    import text.cleaners # Импортируем весь модуль для проверки наличия клинеров
except ImportError as e:
    st.error(f"**Критическая ошибка: Не удалось импортировать модули Tacotron2.**")
    st.error(f"Детали: {e}")
    st.error("Пожалуйста, убедитесь, что вы находитесь в корневой папке проекта 'Tacotron2' "
             "и ваше виртуальное окружение активировано и содержит все зависимости (файлы hparams.py, model.py и т.д. должны быть доступны).")
    st.stop()

# --- Добавление пути к HiFi-GAN и импорт его зависимостей ---
# Убеждаемся, что путь к hifigan добавлен относительно текущего скрипта
hifigan_path = os.path.join(current_dir, 'hifigan')
if hifigan_path not in sys.path:
    sys.path.append(hifigan_path)

try:
    from hifigan.meldataset import MAX_WAV_VALUE
    from hifigan.models import Generator
    from hifigan.env import AttrDict
except ImportError as e:
    st.error(f"**Критическая ошибка: Не удалось импортировать модули HiFi-GAN.**")
    st.error(f"Детали: {e}")
    st.error(f"Убедитесь, что директория 'hifigan' существует в '{current_dir}' и содержит необходимые файлы (__init__.py, models.py и т.д.).")
    st.stop()

# --- Вспомогательные функции ---

def plot_data(st_element, data_list, titles=None, figsize=(16, 4)):
    """
    Отображает данные (спектрограммы, выравнивания) с помощью matplotlib в Streamlit.
    Принимает список numpy-массивов для отображения.
    """
    if not data_list:
        st_element.warning("Нет данных для отображения графиков.")
        return

    valid_data = [d for d in data_list if isinstance(d, np.ndarray)]
    if not valid_data:
        st_element.warning("Получены некорректные данные для графиков (возможно, None).")
        return

    try:
        num_plots = len(valid_data)
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        if num_plots == 1:
            axes = [axes] # Делаем массив из одного элемента для единообразия

        for i in range(num_plots):
            im = axes[i].imshow(valid_data[i], aspect='auto', origin='lower', interpolation='none')
            if titles and i < len(titles):
                axes[i].set_title(titles[i])
            # fig.colorbar(im, ax=axes[i]) # Раскомментируйте, если нужна цветовая шкала

        plt.tight_layout() # Автоматически подгоняет размеры для лучшего вида

        # Сохраняем во временный файл и отображаем
        img_path = os.path.join(current_dir, 'temp_plot.png')
        plt.savefig(img_path)
        plt.close(fig) # Закрываем фигуру, чтобы освободить память

        image = Image.open(img_path)
        st_element.image(image, use_column_width=True)

        # Удаляем временный файл
        try:
            os.remove(img_path)
        except OSError as e_rem:
            st_element.warning(f"Не удалось удалить временный файл графика: {e_rem}")

    except Exception as e:
        st_element.error(f"Ошибка при построении графика: {e}")
        # st.exception(e) # Раскомментируйте для полного трейсбека в интерфейсе


@st.cache_resource(show_spinner="Загрузка Tacotron2...") # Используем cache_resource для моделей
def load_tts_model(checkpoint_path, _hparams):
    """
    Загружает модель Tacotron2 из чекпоинта.
    Использует кеширование Streamlit для ускорения повторных запусков.
    """
    st.info(f"Загрузка Tacotron2: {os.path.basename(checkpoint_path)}")
    if not os.path.isfile(checkpoint_path):
        st.error(f"**Ошибка:** Файл чекпоинта Tacotron2 НЕ НАЙДЕН по пути: {checkpoint_path}")
        st.stop()

    # Определяем устройство и FP16
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    use_fp16 = _hparams.fp16_run and use_cuda

    try:
        model = load_model(_hparams)

        # Загружаем чекпоинт сначала на CPU, чтобы избежать проблем с памятью GPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Определяем state_dict (может быть внутри 'state_dict' или напрямую)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint: # Иногда бывает 'model'
             state_dict = checkpoint['model'].state_dict()
        else:
            state_dict = checkpoint
            st.warning(f"Ключ 'state_dict' или 'model' не найден в чекпоинте '{os.path.basename(checkpoint_path)}'. Попытка загрузить основной объект как state_dict.")

        # Убираем префикс 'module.', если модель была сохранена из DistributedDataParallel
        # Создаем новый словарь, чтобы не изменять оригинальный state_dict на всякий случай
        new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        state_dict = new_state_dict

        # Загружаем веса (strict=False позволяет загружать частично, если есть несовпадения)
        load_result = model.load_state_dict(state_dict, strict=False)

        # Выводим информацию о несовпадениях ключей - полезно для отладки
        if load_result.missing_keys:
            st.warning(f"**Предупреждение (Tacotron2 Missing Keys):** Модель ожидает эти веса, но их нет в чекпоинте: `{load_result.missing_keys}`.")
        if load_result.unexpected_keys:
            st.warning(f"**Предупреждение (Tacotron2 Unexpected Keys):** Эти веса есть в чекпоинте, но не ожидаются моделью: `{load_result.unexpected_keys}`.")

        # Переносим модель на нужное устройство и устанавливаем режим eval()
        model = model.to(device)
        if use_fp16:
            model.half()
            st.info("Tacotron2 работает в режиме FP16 (на GPU).")
        model.eval() # Переводим модель в режим инференса

        if not use_cuda:
            st.warning("CUDA недоступна. Tacotron2 будет работать на CPU (может быть медленно).")
            if _hparams.fp16_run:
                st.warning("FP16 включен в настройках, но CUDA недоступна. FP16 требует GPU.")

        st.success(f"Модель Tacotron2 успешно загружена ({'GPU' if use_cuda else 'CPU'}).")
        return model
    except Exception as e:
        st.error(f"**Критическая ошибка при загрузке модели Tacotron2 из {checkpoint_path}:**")
        st.exception(e) # Показываем полный трейсбек в интерфейсе
        st.stop()


@st.cache_resource(show_spinner="Загрузка HiFi-GAN...") # Используем cache_resource для моделей
def load_vocoder_model(config_rel_path="hifigan/UNIVERSAL_V1/config.json", checkpoint_rel_path="hifigan/UNIVERSAL_V1/g_02500000"):
    """
    Загружает модель вокодера HiFi-GAN.
    Пути указываются относительно директории скрипта.
    """
    # Строим абсолютные пути
    config_path = os.path.join(current_dir, config_rel_path)
    checkpoint_path = os.path.join(current_dir, checkpoint_rel_path)

    st.info(f"Загрузка HiFi-GAN: {os.path.basename(checkpoint_path)}")
    if not os.path.isfile(config_path):
        st.error(f"**Ошибка:** Файл конфигурации HiFi-GAN НЕ НАЙДЕН: {config_path}")
        st.stop()
    if not os.path.isfile(checkpoint_path):
        st.error(f"**Ошибка:** Файл чекпоинта HiFi-GAN НЕ НАЙДЕН: {checkpoint_path}")
        st.stop()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Загружаем конфиг
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        h_vocoder = AttrDict(json_config) # Превращаем словарь в объект с доступом через точку

        # Создаем модель генератора
        generator = Generator(h_vocoder).to(device)

        # Загружаем чекпоинт вокодера
        # Сначала на CPU, чтобы избежать проблем с памятью GPU
        state_dict_g = torch.load(checkpoint_path, map_location='cpu')

        # Загружаем state_dict (может быть 'generator' или напрямую)
        if 'generator' in state_dict_g:
            generator.load_state_dict(state_dict_g['generator'])
        else:
            generator.load_state_dict(state_dict_g)
            st.warning("Ключ 'generator' не найден в чекпоинте вокодера, загружен основной объект.")

        # Переводим в режим инференса и убираем нормализацию весов (важно для HiFi-GAN)
        generator.eval()
        generator.remove_weight_norm()

        # Переносим на нужное устройство (если это не CPU)
        if device.type == 'cuda':
            generator = generator.to(device)

        st.success(f"Вокодер HiFi-GAN успешно загружен ({device.type.upper()}).")
        # Возвращаем и генератор, и его параметры (h_vocoder), т.к. там может быть sample_rate
        return generator, h_vocoder

    except Exception as e:
        st.error(f"**Критическая ошибка при загрузке вокодера HiFi-GAN из {checkpoint_path}:**")
        st.exception(e)
        st.stop()


def inference_hifigan(mel_spectrogram, generator):
    """
    Преобразует mel-спектрограмму в аудио с помощью вокодера HiFi-GAN.
    Вход: mel_spectrogram - тензор PyTorch [1, n_mel, length]
    Выход: numpy массив с аудиоданными [audio_length]
    """
    if mel_spectrogram is None:
        st.error("Ошибка: Инференс вокодера получил пустую mel-спектрограмму (None).")
        return None
    if not isinstance(mel_spectrogram, torch.Tensor):
         st.error(f"Ошибка: Инференс вокодера ожидал тензор PyTorch, но получил {type(mel_spectrogram)}.")
         return None

    device = next(generator.parameters()).device # Определяем устройство, на котором находится модель

    try:
        # Убедимся, что входной тензор на нужном устройстве и типа float
        mel_spectrogram = mel_spectrogram.to(device).float()

        with torch.no_grad(): # Отключаем вычисление градиентов для инференса
            y_g_hat = generator(mel_spectrogram) # Инференс вокодера
            audio = y_g_hat.squeeze().cpu().numpy() # Убираем лишние размерности, переносим на CPU, конвертируем в numpy
            # Нормализация или масштабирование здесь не делается, HiFi-GAN обычно выдает [-1, 1]
            # audio = audio * MAX_WAV_VALUE # Эта строка из оригинального кода часто не нужна или зависит от модели
            # audio = audio.astype('int16') # Конвертация в int16 обычно делается при записи в файл
        return audio

    except Exception as e:
        st.error(f"Ошибка во время инференса вокодера HiFi-GAN:")
        st.exception(e)
        return None

# --- Основная функция приложения Streamlit ---

def main():
    # Настройка страницы Streamlit
    st.set_page_config(
        layout="wide",
        page_title="Russian TTS Demo",
        page_icon="🗣️"
    )
    st.title("🗣️ Демо синтеза речи: Tacotron2 + HiFi-GAN")
    st.markdown("Введите текст на русском языке (можно использовать `+` для явного указания ударения перед гласной) и нажмите 'Синтезировать'.")

    # --- Конфигурация в боковой панели ---
    st.sidebar.header("⚙️ Настройки")

    # --- Настройки Tacotron2 ---
    st.sidebar.subheader("1. Модель Tacotron2 (Текст → Мель)")
    try:
        # Создаем базовые гиперпараметры
        hparams = create_hparams()
        # Включаем FP16, только если доступна CUDA (оптимизация)
        hparams.fp16_run = torch.cuda.is_available()
    except Exception as e:
        st.error(f"Ошибка при создании гиперпараметров (hparams): {e}")
        st.stop()

    # Автоматический поиск чекпоинтов в директории 'outdir' рядом со скриптом
    weights_dir = os.path.join(current_dir, "outdir")
    available_checkpoints_paths = []
    default_checkpoint_path = None
    if os.path.isdir(weights_dir):
        try:
            files = os.listdir(weights_dir)
            # Фильтруем файлы: начинаются с 'checkpoint_', могут иметь расширения .pt, .pth или быть без расширения
            checkpoints = [
                os.path.join(weights_dir, f) for f in files
                if os.path.isfile(os.path.join(weights_dir, f)) and
                   f.startswith('checkpoint_') and
                   (f.endswith(('.pth', '.pt')) or '.' not in f.split('_')[-1]) # Простой фильтр
            ]
            # Сортируем (попытка сортировки по номеру итерации, если он есть)
            try:
                # Пытаемся извлечь число после 'checkpoint_'
                checkpoints_sorted = sorted(
                    checkpoints,
                    key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]), # Извлекаем число
                    reverse=True # Последние чекпоинты (с большим номером) первыми
                )
            except (IndexError, ValueError):
                # Если не получилось (формат имени другой), сортируем по имени файла
                checkpoints_sorted = sorted(checkpoints, reverse=True)

            available_checkpoints_paths = checkpoints_sorted
            if available_checkpoints_paths:
                default_checkpoint_path = available_checkpoints_paths[0] # Берем первый (последний по номеру/имени)
                st.sidebar.info(f"Найдено {len(available_checkpoints_paths)} чекпоинтов в '{os.path.basename(weights_dir)}'. Выбран: {os.path.basename(default_checkpoint_path)}")
            else:
                 st.sidebar.warning(f"Директория '{os.path.basename(weights_dir)}' найдена, но чекпоинты (checkpoint_*) не обнаружены.")
        except Exception as e:
            st.sidebar.warning(f"Ошибка при сканировании директории '{weights_dir}': {e}")
    else:
        st.sidebar.warning(f"Директория для чекпоинтов '{os.path.basename(weights_dir)}' не найдена рядом со скриптом. Поместите чекпоинты туда или укажите путь вручную ниже.")

    # Выбор чекпоинта из найденных или ввод вручную
    selected_checkpoint_path = st.sidebar.selectbox(
        "Чекпоинт Tacotron2:",
        options=available_checkpoints_paths,
        format_func=lambda x: os.path.basename(x) if x else "Нет доступных", # Показываем только имя файла
        index=0 if default_checkpoint_path else None, # Выбираем первый по умолчанию, если есть
        key="tacotron_checkpoint_select",
        help="Выберите чекпоинт из найденных в папке 'outdir' или укажите путь ниже."
    )
    custom_path = st.sidebar.text_input(
        'Или укажите ПОЛНЫЙ путь к файлу чекпоинта:',
        value="",
        key="tacotron_checkpoint_custom",
        help="Используйте это поле, если нужный файл не в списке или не в директории 'outdir'."
    )

    # Определение финального пути к чекпоинту
    final_checkpoint_path = None
    potential_path = custom_path.strip() if custom_path.strip() else selected_checkpoint_path
    if potential_path:
        if os.path.isfile(potential_path):
            final_checkpoint_path = potential_path
        else:
            st.sidebar.error(f"Указанный файл НЕ НАЙДЕН: {potential_path}")
    else:
         st.sidebar.error("Чекпоинт Tacotron2 не выбран и не указан.")

    # Если путь так и не определен, останавливаемся
    if not final_checkpoint_path:
        st.error("Необходимо выбрать или указать действительный файл чекпоинта Tacotron2.")
        st.stop()

    # Настройка GST (если поддерживается моделью)
    # Проверяем, есть ли параметр use_gst в hparams по умолчанию
    default_use_gst = getattr(hparams, 'use_gst', False) # Безопасно получаем значение, False если атрибута нет
    use_gst_override = st.sidebar.checkbox("Использовать GST?", value=default_use_gst, key="gst_checkbox",
                                           help=f"Включите, если ваш чекпоинт Tacotron2 обучался с Global Style Tokens. Значение по умолчанию из hparams: {default_use_gst}")
    hparams.use_gst = use_gst_override
    if use_gst_override != default_use_gst:
        st.sidebar.info(f"Настройка GST изменена на `{use_gst_override}` для этого запуска.")

    # --- Настройки HiFi-GAN ---
    st.sidebar.subheader("2. Вокодер HiFi-GAN (Мель → Аудио)")
    # Пути к файлам вокодера (относительно скрипта)
    # Используем модель UNIVERSAL_V1 как пример/дефолт
    hifigan_config_default = "hifigan/UNIVERSAL_V1/config.json"
    hifigan_checkpoint_default = "hifigan/UNIVERSAL_V1/g_02500000"
    # Можно добавить поля для ввода путей к вокодеру, если нужно
    st.sidebar.caption(f"Используется конфиг: `{hifigan_config_default}`")
    st.sidebar.caption(f"Используется чекпоинт: `{hifigan_checkpoint_default}`")

    # --- Настройки Синтеза ---
    st.sidebar.subheader("3. Параметры синтеза")
    # Сид для воспроизводимости
    seed_input = st.sidebar.text_input('Сид генерации (пусто = случайно)', value='', key='seed_input')
    seed_value = None
    if seed_input.strip():
        try:
            seed_value = int(seed_input)
        except ValueError:
            st.sidebar.error("Сид должен быть целым числом!")
            seed_value = None # Сбрасываем, если ошибка

    # Выбор текстового клинера
    # Динамически определяем доступные клинеры из модуля text.cleaners
    available_cleaners = {}
    if hasattr(text.cleaners, 'transliteration_cleaners_with_stress'):
        available_cleaners["Русский с ударениями (+)"] = 'transliteration_cleaners_with_stress'
    if hasattr(text.cleaners, 'transliteration_cleaners'):
        available_cleaners["Русский без ударений"] = 'transliteration_cleaners'
    if hasattr(text.cleaners, 'russian_cleaners'): # Добавим еще один для примера
         available_cleaners["Русский базовый"] = 'russian_cleaners'
    # Добавьте сюда другие клинеры, если они есть в вашем text/cleaners.py

    if not available_cleaners:
        st.error("В модуле `text.cleaners` не найдены подходящие клинеры для русского языка.")
        st.stop()

    # Определяем клинер по умолчанию из hparams, если возможно
    default_cleaner_value = None
    if hparams.text_cleaners and isinstance(hparams.text_cleaners, list):
        default_cleaner_value = hparams.text_cleaners[0] # Берем первый из списка в hparams

    default_cleaner_name = None
    cleaner_names = list(available_cleaners.keys())
    for name, value in available_cleaners.items():
        if value == default_cleaner_value:
            default_cleaner_name = name
            break

    default_cleaner_index = cleaner_names.index(default_cleaner_name) if default_cleaner_name else 0

    # Выбор клинера в интерфейсе
    selected_cleaner_name = st.sidebar.selectbox(
        "Обработка текста:",
        options=cleaner_names,
        index=default_cleaner_index,
        key="cleaner_select",
        help="Выбор способа предварительной обработки текста перед подачей в модель."
    )
    # Получаем фактическое имя клинера для использования
    selected_cleaner_internal_name = [available_cleaners[selected_cleaner_name]]

    # --- Загрузка моделей (после всех настроек) ---
    # Помещаем в expander, чтобы не загромождать интерфейс при старте
    with st.expander("Статус загрузки моделей", expanded=True):
        model = load_tts_model(final_checkpoint_path, hparams)
        vocoder, vocoder_hparams = load_vocoder_model(hifigan_config_default, hifigan_checkpoint_default)

        # Проверка совпадения частоты дискретизации
        tacotron_sr = getattr(hparams, 'sampling_rate', None)
        vocoder_sr = getattr(vocoder_hparams, 'sampling_rate', None)
        if tacotron_sr is not None and vocoder_sr is not None and tacotron_sr != vocoder_sr:
            st.warning(f"**Внимание:** Частота дискретизации Tacotron2 ({tacotron_sr} Гц) "
                       f"не совпадает с HiFi-GAN ({vocoder_sr} Гц). Это может повлиять на качество звука.")
        elif tacotron_sr is None:
             st.warning("Не удалось определить частоту дискретизации из hparams Tacotron2.")

    # --- Интерфейс ввода текста ---
    st.subheader("📝 Введите текст для синтеза")
    # Примеры текста
    predefined_texts = [
        "Н+очь, +улица, фон+арь, апт+ека. Бессм+ысленный и т+усклый св+ет.",
        "мн+е хот+елось бы сказ+ать, как я призн+ателен вс+ем прис+утствующим зд+есь.",
        "Тв+орог или твор+ог? к+озлы или козл+ы? з+амок или зам+ок?",
        "Вс+е смеш+алось в д+оме Обл+онских. Провер+яем дл+инное предлож+ение.",
        "Съ+ешь же ещё эт+их м+ягких франц+узских б+улок да в+ыпей ч+аю.",
        "Тетрагидропиранилциклопентилтетрагидропиридопиридиновые веществ+а - звуч+ат сложнов+ато."
    ]
    text_options = ["(Свой текст)"] + predefined_texts
    selected_text_option = st.selectbox(
        "Выберите пример или введите свой:",
        text_options,
        index=1, # По умолчанию выбираем первый пример
        key="text_select"
    )

    # Поле для ввода текста
    default_text_value = selected_text_option if selected_text_option != "(Свой текст)" else "Прив+ет, мир! Как дел+а?"
    text_input = st.text_area(
        "Текст:",
        value=default_text_value,
        height=100,
        key="text_input_area",
        help="Используйте '+' перед гласной для указания ударения (если выбран клинер с ударениями)."
    )

    st.markdown("---") # Разделитель
    # Кнопка для запуска синтеза
    generate_button = st.button("🔊 Синтезировать Речь", type="primary", key="generate_button")
    st.markdown("---") # Разделитель

    # --- Генерация и отображение результатов ---
    if generate_button and text_input.strip(): # Запускаем только если кнопка нажата и текст не пустой
        st.subheader("📊 Результаты Синтеза")

        # Создаем колонки для отображения
        col1, col2 = st.columns([0.6, 0.4]) # Графики шире, информация/аудио уже

        # Плейсхолдеры для обновления содержимого без полного перерисовывания
        with col1:
            plots_placeholder = st.container()
            plots_placeholder.write("**Визуализация:**") # Заголовок для графиков
        with col2:
            info_placeholder = st.empty() # Для информации о запуске
            audio_placeholder = st.empty() # Для аудио плеера

        with st.spinner("Синтез речи... ⏳"):
            start_time = time.perf_counter()

            # 1. Преобразование текста в последовательность ID
            sequence = None
            try:
                sequence_np = np.array(text_to_sequence(text_input, selected_cleaner_internal_name))[None, :]
                sequence = torch.from_numpy(sequence_np).long()
                # Переносим на GPU, если доступно
                if torch.cuda.is_available():
                    sequence = sequence.cuda()
            except Exception as e:
                st.error(f"**Ошибка при преобразовании текста в последовательность:**")
                st.exception(e)
                st.stop() # Останавливаем выполнение, если текст не обработан

            # 2. Инференс Tacotron2 (Получение Mel-спектрограммы)
            # Инициализируем переменные как None
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = None, None, None, None
            gst_embedding = None # Добавляем инициализацию для переменной из случая 6 выходов
            mel_to_vocoder = None

            try:
                with torch.no_grad(): # Отключаем градиенты
                    # Используем model.inference
                    inference_args = (sequence,)
                    inference_kwargs = {}
                    if hparams.use_gst: # Передаем параметры для GST, если он включен
                         # Здесь можно добавить логику для reference_mel, token_idx, scale, если нужно
                         pass # Пока просто передаем пустой kwargs, если не заданы особые параметры GST

                    if seed_value is not None:
                        import inspect
                        sig = inspect.signature(model.inference)
                        if 'seed' in sig.parameters:
                           inference_kwargs['seed'] = seed_value
                        else:
                           st.warning("Модель Tacotron2 не поддерживает установку сида через метод inference.")

                    outputs = model.inference(*inference_args, **inference_kwargs)

                    # ----- ИСПРАВЛЕННЫЙ БЛОК РАСПАКОВКИ -----
                    if len(outputs) == 4: # Стандартный вывод Tacotron2 Nvidia
                         st.info("Tacotron2 inference вернул 4 значения.")
                         mel_outputs, mel_outputs_postnet, gate_outputs, alignments = outputs
                         gst_embedding = None # Убедимся, что gst_embedding = None в этом случае
                    elif len(outputs) == 5: # Иногда возвращают еще stop_token
                         st.info("Tacotron2 inference вернул 5 значений.")
                         # Предполагаем, что GST здесь не возвращается, а 5-й элемент - stop_token или что-то еще
                         mel_outputs, mel_outputs_postnet, gate_outputs, alignments, _ = outputs
                         gst_embedding = None # Убедимся, что gst_embedding = None в этом случае
                    elif len(outputs) == 6: # ***Случай с 6 выходами из вашего model.py***
                         st.info("Tacotron2 inference вернул 6 значений (согласно model.py).")
                         # КОРРЕКТНАЯ РАСПАКОВКА: None, mel, mel_post, gate, align, gst
                         decoder_outputs_ignored, mel_outputs, mel_outputs_postnet, gate_outputs, alignments, gst_embedding = outputs
                         # Выводим информацию о полученном GST embedding для отладки
                         if isinstance(gst_embedding, torch.Tensor):
                             st.info(f"Получен GST embedding: shape={gst_embedding.shape}, dtype={gst_embedding.dtype}")
                         else:
                             st.warning(f"Получен 6-й элемент (ожидался GST), но это не тензор: type={type(gst_embedding)}")
                    else: # Ошибка, если количество не совпадает ни с одним из ожидаемых
                         st.error(f"Неожиданное количество выходов из model.inference: {len(outputs)}. Ожидалось 4, 5 или 6.")
                         st.stop()
                    # ----- КОНЕЦ ИСПРАВЛЕННОГО БЛОКА -----

                    # Выбираем спектрограмму для вокодера (обычно postnet)
                    if mel_outputs_postnet is not None:
                         mel_to_vocoder = mel_outputs_postnet.float() # Убедимся, что тип float
                    elif mel_outputs is not None:
                         st.warning("Postnet mel-спектрограмма отсутствует, используется mel_outputs для вокодера.")
                         mel_to_vocoder = mel_outputs.float()
                    else:
                         st.error("Tacotron2 не сгенерировал ни mel_outputs, ни mel_outputs_postnet.")
                         st.stop()

            except Exception as e:
                st.error(f"**Ошибка во время инференса Tacotron2:**")
                st.exception(e)
                st.stop() # Останавливаем, если Tacotron не отработал

            # 3. Инференс HiFi-GAN (Получение аудио из Mel)
            audio = None
            if mel_to_vocoder is not None: # Проверяем, что есть что подавать вокодеру
                try:
                    audio = inference_hifigan(mel_to_vocoder, vocoder)
                except Exception as e:
                    # Ошибка уже должна быть обработана внутри inference_hifigan, но на всякий случай
                    st.error(f"**Ошибка во время инференса вокодера HiFi-GAN:**")
                    st.exception(e)
                    st.stop() # Останавливаем, если вокодер упал
            else:
                 st.error("Нет mel-спектрограммы для передачи в вокодер.")
                 st.stop()


            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # --- Отображение результатов ---

            # Информация о запуске
            with info_placeholder.container():
                st.success(f"🎉 Речь сгенерирована за **{elapsed_time:.2f} сек.**")
                st.markdown("**Параметры запуска:**")
                st.markdown(f"- **Чекпоинт:** `{os.path.basename(final_checkpoint_path)}`")
                st.markdown(f"- **Клинер:** `{selected_cleaner_name}` (`{selected_cleaner_internal_name[0]}`)")
                st.markdown(f"- **GST:** `{hparams.use_gst}` | **Сид:** `{seed_value if seed_value is not None else 'Случайный'}`")
                st.markdown(f"- **Устройство:** `{'GPU' if torch.cuda.is_available() else 'CPU'}`")
                # Выводим информацию о GST эмбеддинге, если он был получен
                if gst_embedding is not None and isinstance(gst_embedding, torch.Tensor):
                    # Показываем только shape и dtype, так как сам тензор может быть большим
                    st.markdown(f"- **GST Embedding Shape:** `{gst_embedding.shape}`")


            # Графики (если данные получены)
            with plots_placeholder: # Используем ранее созданный контейнер
                data_for_plot = []
                plot_titles = []
                if mel_outputs is not None:
                    # Используем .data, чтобы избежать проблем с градиентами при конвертации в numpy
                    data_for_plot.append(mel_outputs.float().data.cpu().numpy()[0])
                    plot_titles.append("Mel (до postnet)")
                if mel_outputs_postnet is not None:
                     data_for_plot.append(mel_outputs_postnet.float().data.cpu().numpy()[0])
                     plot_titles.append("Mel (после postnet)")
                if alignments is not None:
                     data_for_plot.append(alignments.float().data.cpu().numpy()[0].T) # Транспонируем для лучшего вида
                     plot_titles.append("Выравнивание")

                if data_for_plot:
                     plot_data(st, data_for_plot, titles=plot_titles, figsize=(10, 3)) # Уменьшим размер
                else:
                     st.warning("Не удалось получить данные для построения графиков.")


            # Аудио (если сгенерировано)
            with audio_placeholder.container(): # Используем ранее созданный контейнер
                if audio is not None:
                    st.write("**Аудио результат:**")
                    output_wav_path = os.path.join(current_dir, "output_audio.wav")
                    audio_sr = vocoder_sr if vocoder_sr else 22050 # Используем SR вокодера или дефолт
                    try:
                        # Нормализуем аудио к диапазону PCM16 перед записью, если оно в [-1, 1]
                        if np.abs(audio).max() <= 1.0:
                            audio_int16 = (audio * 32767).astype(np.int16)
                        else:
                            # Если аудио уже в другом диапазоне, пытаемся записать как есть
                            st.warning("Аудио из вокодера имеет диапазон > 1.0. Запись без нормализации.")
                            audio_int16 = audio.astype(np.int16) # Попытка конвертации

                        sf.write(output_wav_path, audio_int16, audio_sr, subtype='PCM_16')

                        # Читаем байты для отображения и скачивания
                        with open(output_wav_path, 'rb') as audio_file:
                            audio_bytes = audio_file.read()

                        st.audio(audio_bytes, format='audio/wav', sample_rate=audio_sr)
                        st.download_button(
                            label="📥 Скачать WAV",
                            data=audio_bytes,
                            file_name="generated_speech.wav",
                            mime='audio/wav'
                        )
                    except Exception as e:
                        st.error(f"**Ошибка при сохранении или отображении аудио:**")
                        st.exception(e)
                    finally:
                        # Удаляем временный файл, даже если были ошибки
                        if os.path.exists(output_wav_path):
                            try:
                                os.remove(output_wav_path)
                            except OSError as e_rem:
                                st.warning(f"Не удалось удалить временный аудиофайл {output_wav_path}: {e_rem}")
                else:
                    st.error("Аудио не было сгенерировано.")

    elif generate_button and not text_input.strip():
        st.warning("⚠️ Пожалуйста, введите текст для синтеза.")

# --- Точка входа для запуска скрипта ---
if __name__ == "__main__":
    # Устанавливаем seed для PyTorch и NumPy для большей воспроизводимости, если он задан
    # Делаем это один раз при старте, если нужно глобально, но лучше передавать в inference
    # if seed_value is not None:
    #     torch.manual_seed(seed_value)
    #     np.random.seed(seed_value)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed_value)
    #         torch.backends.cudnn.deterministic = True # Может замедлить, но улучшает воспроизводимость
    #         torch.backends.cudnn.benchmark = False

    main()
