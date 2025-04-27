from text import symbols
# Убедитесь, что класс HParams импортирован правильно для вашего проекта
from tools import HParams # Используем ваш импорт

# Можно использовать стандартный logging
import logging

def create_hparams(hparams_string=None, verbose=False):
    """Создает гиперпараметры модели. Парсит нестандартные параметры из строки."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500000, # НЕ РЕАЛИСТИЧНАЯ ЦЕЛЬ. Обучение остановится раньше (вручную или по валидации).
                       # Фактический прогресс отслеживается по ИТЕРАЦИЯМ (шагам).
                       # Типичный диапазон для сходимости: 100k - 300k+ ИТЕРАЦИЙ.
                       # Установите значение ~500k-1M, чтобы точно не прерваться по эпохам.
        iters_per_checkpoint=5000, # Сохраняем чекпоинт каждые 2000 шагов.
                                   # Можно начать с 1000, потом увеличить до 5000 для экономии места.
        seed=1234,                 # Для воспроизводимости экспериментов
        dynamic_loss_scaling=True, # Обязательно True для FP16
        fp16_run=True,             # !!! ВКЛЮЧЕНО: Используем Mixed Precision для ускорения на RTX 4070 Ti SUPER !!!
        distributed_run=False,     # Обучение на одной GPU
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,        # Включаем cuDNN
        cudnn_benchmark=True,      # Может дать небольшое ускорение, если размеры входов стабильны (что обычно так)

        # Слои для игнорирования при загрузке чекпоинта (если нужно)
        ignore_layers=['embedding.weight'],
        # Слои для игнорирования при MMI обучении (если используется)
        mmi_ignore_layers=["decoder.linear_projection.linear_layer.weight", "decoder.linear_projection.linear_layer.bias", "decoder.gate_layer.linear_layer.weight"],

        ################################
        # Data Parameters              #
        ################################
        load_mel_from_disk=True,
        dataset_path="data", # Убедитесь, что это правильный путь к аудио
        # !!! РАЗДЕЛИТЕ ДАННЫЕ: Укажите ОТДЕЛЬНЫЕ файлы для обучения и валидации !!!
        training_files="data/train.csv", # Файл ТОЛЬКО с обучающими данными
        validation_files="data/validation.csv", # Файл ТОЛЬКО с валидационными данными (например, 5-10% от общих)
        text_cleaners=['transliteration_cleaners_with_stress'],

        ################################
        # Audio Parameters             #
        ################################
        # Стандартные параметры, совместимые с HiFi-GAN Universal (22050 Гц)
        max_wav_value=32768.0,    # Макс. значение для int16 WAV
        sampling_rate=22050,      # Частота дискретизации
        filter_length=1024,       # Длина FFT окна
        hop_length=256,           # Шаг окна (определяет временное разрешение Mel) -> ~11.6 мс на кадр
        win_length=1024,          # Длина окна Ханна
        n_mel_channels=80,        # Количество Mel-фильтров
        mel_fmin=0.0,             # Минимальная частота для Mel-фильтров
        mel_fmax=8000.0,          # Максимальная частота (рекомендуется < sampling_rate / 2)

        ################################
        # Model Parameters             #
        ################################
        # Размерности модели - стандартные значения для Tacotron 2.
        # Увеличение может улучшить качество на больших данных, но требует больше VRAM и замедляет обучение.
        # Для 5.7 часов данных эти размеры должны быть достаточны.
        n_symbols=len(symbols),             # Количество уникальных символов в вашем алфавите (из text/symbols.py)
        symbols_embedding_dim=512,          # Размер эмбеддинга символов

        # Encoder
        encoder_kernel_size=5,              # Размер сверточного ядра в энкодере
        encoder_n_convolutions=3,           # Количество сверточных слоев в энкодере
        encoder_embedding_dim=512,          # Выходная размерность энкодера (должна быть равна symbols_embedding_dim)

        # Decoder
        n_frames_per_step=1,                # Количество Mel-кадров, генерируемых за один шаг декодера (1 - лучшее качество)
        decoder_rnn_dim=1024,               # Размер скрытого состояния LSTM в декодере
        prenet_dim=256,                     # Размерность слоев Prenet
        max_decoder_steps=1500,             # Максимальное кол-во шагов декодера (ограничивает макс. длину аудио ~17 сек)
        gate_threshold=0.5,                 # Порог для stop-токена (предсказание конца генерации)
        p_attention_dropout=0.1,            # Dropout для attention LSTM
        p_decoder_dropout=0.1,              # Dropout для decoder LSTM
        p_teacher_forcing=1.0,              # Вероятность использования истинного Mel-кадра (teacher forcing). 1.0 - стандарт для начала.

        # Attention
        attention_rnn_dim=1024,             # Размер attention LSTM (часто равен decoder_rnn_dim)
        attention_dim=128,                  # Внутренняя размерность механизма внимания

        # Location Sensitive Attention
        attention_location_n_filters=32,    # Кол-во фильтров в location-слое
        attention_location_kernel_size=31,  # Размер ядра в location-слое

        # Postnet (улучшает детализацию Mel-спектрограммы)
        postnet_embedding_dim=512,          # Размерность в Postnet
        postnet_kernel_size=5,              # Размер ядра сверток в Postnet
        postnet_n_convolutions=5,           # Количество сверточных слоев в Postnet

        # GST (Global Style Tokens) - для контроля просодии (если включено)
        use_gst=True,                      # Использовать ли GST
        # Стандартные параметры GST. Менять только при необходимости.
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,
        token_embedding_size=256, # Обычно encoder_embedding_dim // num_heads
        token_num=10,             # Количество токенов стиля
        num_heads=8,              # Количество "голов" в MultiHeadAttention для GST
        # !!! УБЕДИТЕСЬ, ЧТО В train.py ИСПОЛЬЗУЕТСЯ Diagonal Guided Attention (DGA) loss !!!
        # Этот механизм сильно помогает сходимости выравнивания.
        # Параметр no_dga здесь может не влиять, если DGA реализован в функции потерь.
        no_dga=False,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False, # Ставьте True, только если продолжаете обучение с чекпоинта и хотите сохранить LR из него
        learning_rate=1e-3,          # НАЧАЛЬНАЯ скорость обучения.
                                     # !!! ВОЗМОЖНО, ПОТРЕБУЕТСЯ УМЕНЬШИТЬ (до 5e-4, 2e-4, 1e-4) !!!
                                     # если выравнивание (alignment) долго не сходится (остается размытым).
                                     # Рассмотрите использование LR Schedulers (часто настраиваются в train.py).
        weight_decay=1e-6,           # L2 регуляризация (предотвращает переобучение)
        grad_clip_thresh=1.0,        # Обрезка градиента (предотвращает "взрыв" градиентов)
        # !!! ОПТИМИЗИРОВАННЫЙ batch_size для 16GB VRAM + FP16 !!!
        batch_size=48,               # НАЧАЛЬНОЕ ЗНАЧЕНИЕ. Мониторьте VRAM!
                                     # Если память используется < 80%, можно ПОПРОБОВАТЬ УВЕЛИЧИТЬ (56, 64...).
                                     # Если ошибка "Out of Memory", УМЕНЬШАЙТЕ (40, 32...).
        mask_padding=True,           # Использовать маскирование для последовательностей разной длины в батче (обязательно)

        ################################
        # FINE-TUNE / ADVANCED Params  #
        ################################
        # Эти параметры обычно используются для дообучения или специфичных техник
        use_mmi=False,
        drop_frame_rate=0.0,
        use_gaf=False,
        update_gaf_every_n_step=10,
        max_gaf=0.5,
        global_mean_npy=None, # Путь к .npy файлу с глобальным средним/стандартным отклонением Mel (если используется)
    )

    if hparams_string:
        logging.info(f'Parsing command line hparams: {hparams_string}')
        hparams.parse(hparams_string)

    if verbose:
        # Преобразуем значения в строку для логирования
        hparams_values_str = str(hparams.values())
        logging.info(f'Final parsed hparams: {hparams_values_str}')

    return hparams
