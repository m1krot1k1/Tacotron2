import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read # Используется scipy для чтения wav
from librosa.filters import mel as librosa_mel_fn

# Импорт AttrDict, если он используется здесь (хотя обычно он только в train.py/env.py)
# from env import AttrDict # Если AttrDict нужен здесь, раскомментируйте


MAX_WAV_VALUE = 32768.0

# Используется scipy.io.wavfile.read
def load_wav(full_path):
    """
    Загружает WAV файл с использованием scipy.
    """
    try:
        sampling_rate, data = read(full_path)
        # Проверка типа данных scipy read - может вернуть int16, int32, float32
        if data.dtype != np.float32:
            # Преобразуем в float32 и нормализуем
            data = data.astype(np.float32) / MAX_WAV_VALUE
        return data, sampling_rate
    except Exception as e:
        print(f"Ошибка при загрузке WAV файла {full_path}: {e}")
        # Возвращаем пустые данные и 0 rate в случае ошибки
        return np.array([], dtype=np.float32), 0


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


# Глобальные кэши для ускорения
mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    Вычисляет мел-спектрограмму для батча аудио.
    """
    # y: (Batch, Audio_samples)

    # Проверка диапазонов аудио (опционально, для отладки)
    # if torch.min(y) < -1.001 or torch.max(y) > 1.001: # Добавлен небольшой допуск
    #      print(f'Предупреждение: Значения аудио выходят за пределы [-1, 1]. Min: {torch.min(y)}, Max: {torch.max(y)}')


    global mel_basis, hann_window
    # Кэшируем mel_basis и hann_window по fmax и устройству
    key = f"{fmax}_{str(y.device)}"
    if key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)

    device_key = str(y.device)
    if device_key not in hann_window or hann_window[device_key].size(0) != win_size: # Добавлена проверка размера окна
         hann_window[device_key] = torch.hann_window(win_size).to(y.device)


    # Добавляем паддинг для STFT
    y = y.unsqueeze(1) # (Batch, 1, Audio_samples)
    # Паддинг: (Batch, 1, Audio_samples + 2 * pad_size)
    y = torch.nn.functional.pad(y, (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1) # (Batch, Audio_samples + 2 * pad_size)


    # Выполнение STFT
    spec = torch.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[device_key],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,
                      return_complex=False) # Указываем return_complex=False для старых версий PyTorch


    # Вычисление магнитуды спектра
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9)) # (Batch, n_freq, num_frames)

    # Применение мел-фильтров
    spec = torch.matmul(mel_basis[key], spec) # (Batch, num_mels, num_frames)

    # Спектральная нормализация
    spec = spectral_normalize_torch(spec) # (Batch, num_mels, num_frames)

    return spec


# Измененная функция get_dataset_filelist
def get_dataset_filelist(a):
    """
    Читает списки файлов из .txt/.csv и формирует полные пути к WAV-файлам.
    Берет только имя файла из списка, игнорируя пути поддиректорий в списке.

    Args:
        a (argparse.Namespace): Объект с аргументами командной строки,
                                включая input_training_file, input_validation_file,
                                и input_wavs_dir (базовая директория с фактическими WAV файлами).

    Returns:
        tuple: Списки полных путей к тренировочным и валидационным WAV файлам.
    """
    # Путь к директории с фактическими WAV-файлами берется из a.input_wavs_dir
    # Это должен быть путь, который вы передаете через аргумент --input_wavs_dir в train.py (например, "../data/wavs")
    wavs_base_dir = a.input_wavs_dir

    print(f"Чтение тренировочного списка файлов из: {a.input_training_file}")
    print(f"Ожидаемая базовая директория для поиска WAV: {wavs_base_dir}")

    training_files = []
    try:
        with open(a.input_training_file, 'r', encoding='utf-8') as fi:
            for line in fi:
                 parts = line.strip().split('|')
                 if len(parts) > 0 and len(parts[0]) > 0:
                     # parts[0] - это строка из CSV, например, "LJSpeech-1.1/wavs/1_chunk_007.wav"
                     # Используем os.path.basename, чтобы получить только имя файла "1_chunk_007.wav"
                     filename_only = os.path.basename(parts[0])
                     # Объединяем базовую директорию для WAV с только именем файла
                     full_path = os.path.join(wavs_base_dir, filename_only)
                     training_files.append(full_path)

    except FileNotFoundError:
         print(f"Ошибка: Тренировочный файл списка датасета не найден: {a.input_training_file}")
         exit(1) # Выходим, так как без списка обучение невозможно
    except Exception as e:
         print(f"Ошибка при чтении тренировочного файла списка {a.input_training_file}: {e}")
         exit(1)


    print(f"Чтение валидационного списка файлов из: {a.input_validation_file}")
    validation_files = []
    try:
        with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
             for line in fi:
                 parts = line.strip().split('|')
                 if len(parts) > 0 and len(parts[0]) > 0:
                      # parts[0] - это строка из CSV, например, "LJSpeech-1.1/wavs/1_chunk_007.wav"
                      # Используем os.path.basename, чтобы получить только имя файла "1_chunk_007.wav"
                      filename_only = os.path.basename(parts[0])
                      # Объединяем базовую директорию для WAV с только именем файла
                      full_path = os.path.join(wavs_base_dir, filename_only)
                      validation_files.append(full_path)

    except FileNotFoundError:
        print(f"Предупреждение: Валидационный файл списка датасета не найден: {a.input_validation_file}. Валидация будет пропущена.")
        validation_files = [] # Убедимся, что список валидации пуст
    except Exception as e:
         print(f"Ошибка при чтении валидационного файла списка {a.input_validation_file}: {e}")
         print("Валидация будет пропущена из-за ошибки чтения файла.")
         validation_files = [] # Сбрасываем список валидации при ошибке чтения


    # Optional: Добавить проверку существования нескольких первых файлов после формирования путей
    print(f"Проверка существования первых 5 тренировочных файлов в {wavs_base_dir}...")
    files_to_check = training_files[:5]
    if not files_to_check:
         print("Список тренировочных файлов пуст после чтения.")
    else:
        for i, f in enumerate(files_to_check):
            if not os.path.exists(f):
                print(f"ПРЕДУПРЕЖДЕНИЕ: Тренировочный WAV файл не найден по пути: {f}")
            # else:
            #     print(f"ОК: Найден {f}") # Для детальной отладки

    if validation_files:
        print(f"Проверка существования первых 5 валидационных файлов в {wavs_base_dir}...")
        files_to_check_val = validation_files[:5]
        if not files_to_check_val:
             print("Список валидационных файлов пуст после чтения.")
        else:
            for i, f in enumerate(files_to_check_val):
                if not os.path.exists(f):
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Валидационный WAV файл не найден по пути: {f}")
                # else:
                #      print(f"ОК: Найден {f}") # Для детальной отладки
    else:
        print("Список валидационных файлов пуст, проверка пропущена.")


    return training_files, validation_files

class MelDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset для загрузки аудио и мел-спектрограмм для HiFi-GAN.
    """
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files # Список ПОЛНЫХ путей к WAV файлам
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path # Путь к директории с мел-спектрограммами (для fine-tuning)


    def __getitem__(self, index):
        """
        Загружает данные для одного элемента датасета.
        """
        # filename здесь - это ПОЛНЫЙ путь к WAV файлу, сформированный в get_dataset_filelist
        filename = self.audio_files[index]

        # --- Загрузка аудио ---
        # Если не используем кэш или счетчик кэша сброшен, загружаем аудио
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename) # load_wav теперь в этом же файле и использует scipy

            # Проверка на успешную загрузку
            if audio.size == 0 or sampling_rate == 0:
                 print(f"Предупреждение: Пропущен файл из-за ошибки загрузки или пустого содержимого: {filename}")
                 # Возвращаем None или вызываем ошибку, в зависимости от желаемого поведения
                 # Для простоты, попробуем загрузить следующий файл
                 # В реальном коде DataLoader нужно настроить обработку ошибок или None
                 # Пока просто выведем предупреждение и продолжим ( DataLoader может упасть)
                 # return self.__getitem__((index + 1) % len(self)) # Пример рекурсивного вызова (осторожно с бесконечными циклами)
                 raise RuntimeError(f"Не удалось загрузить аудио файл: {filename}") # Лучше вызвать ошибку, чтобы DataLoader упал

            audio = torch.FloatTensor(audio) # Преобразование в тензор

            # Нормализация аудио
            audio = audio / MAX_WAV_VALUE # Нормализация к [-1, 1] на основе MAX_WAV_VALUE
            # if not self.fine_tuning: # Условная нормализация (возможно, уже сделано выше)
            #    audio = normalize(audio.numpy()) * 0.95 # librosa normalize работает с numpy
            #    audio = torch.FloatTensor(audio) # Обратно в тензор


            self.cached_wav = audio # Кэшируем загруженное аудио
            if sampling_rate != self.sampling_rate:
                 print(f"Предупреждение: Частота дискретизации файла {filename} ({sampling_rate} Hz) не совпадает с целевой ({self.sampling_rate} Hz).")
                 # В зависимости от датасета и требований, возможно, нужно ресэмплировать
                 # или пропустить файл. Сейчас просто выводим предупреждение.
                 # raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate)) # Можно раскомментировать для строгой проверки


            self._cache_ref_count = self.n_cache_reuse # Сброс счетчика кэша
        else:
            audio = self.cached_wav # Используем закэшированное аудио
            self._cache_ref_count -= 1 # Уменьшаем счетчик кэша

        # --- Обработка аудио и мел-спектрограмм ---
        audio = audio.unsqueeze(0) # (1, Audio_samples)

        if not self.fine_tuning:
             # В режиме обучения с нуля, генерируем мел из аудио
             if self.split: # Если нужно нарезать сегменты
                 if audio.size(1) >= self.segment_size:
                      max_audio_start = audio.size(1) - self.segment_size
                      audio_start = random.randint(0, max_audio_start)
                      audio = audio[:, audio_start:audio_start+self.segment_size]
                 else:
                      # Паддинг аудио, если оно короче сегмента
                      audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

             # Вычисление мел-спектрограммы из (возможно, обрезанного/паддингованного) аудио
             mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                   center=False)
             # В режиме обучения с нуля mel_loss часто совпадает с mel или вычисляется с другим fmax


        else: # fine_tuning is True
             # В режиме fine-tuning, загружаем предварительно сгенерированные мел-спектрограммы
             # Путь к файлу мел-спектрограммы формируется на основе имени WAV файла и base_mels_path
             # filename - это ПОЛНЫЙ путь к WAV файлу (e.g., ".../data/wavs/1_chunk_007.wav")
             # os.path.basename(filename) - имя файла WAV с расширением (e.g., "1_chunk_007.wav")
             # <-- Убедитесь, что эта строка присутствует и имеет правильный отступ
             wav_basename = os.path.basename(filename)
             # Добавляем .npy к имени файла WAV, чтобы получить "1_chunk_007.wav.npy"
             # <-- Убедитесь, что эта строка имеет такой же отступ, как и строка выше
             mel_filename = os.path.join(self.base_mels_path, wav_basename + '.npy') # <<< ДОЛЖНО ИСПОЛЬЗОВАТЬ wav_basename ЗДЕСЬ

             try:
                 mel = np.load(mel_filename)
             except FileNotFoundError:
                 print(f"Ошибка FileNotFoundError при загрузке файла мел-спектрограммы: {mel_filename}")
                 # В режиме fine-tuning отсутствие мела для WAV файла - это ошибка
                 raise RuntimeError(f"Не удалось загрузить мел-спектрограмму: {mel_filename}") # Вызываем ошибку
             except Exception as e: # Добавлена обработка других возможных ошибок при загрузке numpy
                  print(f"Ошибка при загрузке файла мел-спектрограммы {mel_filename}: {e}")
                  raise RuntimeError(f"Ошибка при загрузке мел-спектрограммы: {mel_filename}") from e # Вызываем ошибку

             mel = torch.from_numpy(mel) # Преобразование numpy в тензор

             if len(mel.shape) < 3:
                 mel = mel.unsqueeze(0) # Добавляем размерность батча/канала, если отсутствует (1, num_mels, num_frames)

             if self.split: # Если нужно нарезать сегменты (при fine-tuning нарезаем по мелам)
                 frames_per_seg = math.ceil(self.segment_size / self.hop_size) # Размер сегмента в мел-фреймах

                 if mel.size(2) >= frames_per_seg: # Проверяем, достаточно ли фреймов в меле
                      mel_start = random.randint(0, mel.size(2) - frames_per_seg) # Случайное начало сегмента мел
                      mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                      # Обрезаем аудио, соответствующее сегменту мел
                      # Начало аудио сегмента = mel_start * hop_size
                      # Конец аудио сегмента = (mel_start + frames_per_seg) * hop_size
                      audio_start = mel_start * self.hop_size
                      audio_end = (mel_start + frames_per_seg) * self.hop_size
                      audio = audio[:, audio_start:audio_end]

                 else:
                      # Паддинг мел-спектрограммы, если она короче сегмента
                      mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                      # Паддинг аудио, чтобы соответствовать длине мел
                      audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        # --- Вычисление мел-спектрограммы для потери (loss) ---
        # Это мел, который будет сравниваться с выходом генератора
        # Обычно используется та же конфигурация, но может быть другой fmax (fmax_for_loss)
        # Вычисляется из (возможно, обрезанного/паддингованного) аудио сегмента
        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                     self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                     center=False)

        # --- Возвращаемые значения ---
        # Возвращаем:
        # 1. Мел-спектрограмма (для входа в генератор, может быть загружена или сгенерирована)
        # 2. Соответствующий сегмент аудио
        # 3. Исходное имя файла (полный путь к WAV)
        # 4. Мел-спектрограмма для вычисления потери (обычно с fmax_for_loss)
        #squeeze(0) убирает размерность батча=1, squeeze() убирает все размерности=1
        return (mel.squeeze(0), audio.squeeze(0), filename, mel_loss.squeeze(0))


    def __len__(self):
        """
        Возвращает общее количество аудиофайлов в датасете.
        """
        return len(self.audio_files)
