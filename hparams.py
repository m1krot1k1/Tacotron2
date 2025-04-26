from tools import HParams
from text import symbols
import tensorflow as tf # Импорт TensorFlow для использования tf.logging

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=5000000, # Максимальное количество эпох, скорее всего, не будет достигнуто
        iters_per_checkpoint=1000, # Сохраняем реже после стабилизации (можно начать с 500, потом увеличить)
        seed=1234,
        dynamic_loss_scaling=True, # Оставляем True, важно для FP16
        fp16_run=True, # !!! ВКЛЮЧАЕМ Mixed Precision для ускорения на вашей карте !!!
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=True, # !!! Устанавливаем True для возможного ускорения, т.к. размеры входов примерно постоянны !!!
        ignore_layers=['embedding.weight'],
        mmi_ignore_layers=["decoder.linear_projection.linear_layer.weight", "decoder.linear_projection.linear_layer.bias", "decoder.gate_layer.linear_layer.weight"],

        ################################
        # Data Parameters              #
        ################################
        load_mel_from_disk=False,
        dataset_path="data", # Убедитесь, что это правильный путь к аудио
        # !!! РАЗДЕЛИТЕ ДАННЫЕ: Укажите ОТДЕЛЬНЫЕ файлы для обучения и валидации !!!
        training_files="data/train.csv", # Файл ТОЛЬКО с обучающими данными
        validation_files="data/validation.csv", # Файл ТОЛЬКО с валидационными данными (например, 5-10% от общих)
        text_cleaners=['transliteration_cleaners_with_stress'],

        ################################
        # Audio Parameters             #
        ################################
        # Эти параметры уже совместимы с HiFi-GAN Universal, оставляем их
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1500, # Можно увеличить (например, до 2000), если есть очень длинные фразы
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        p_teacher_forcing=1.0, # Оставляем 1.0 на начальном этапе

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # GST ======================
        use_gst=True, # Оставляем True, НО см. комментарий ниже
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,
        token_embedding_size=256,
        token_num=10,
        num_heads=8,
        no_dga=False, # Убедитесь, что DGA (Diagonal Guided Attention) используется в вашем train.py (часто это часть лосс-функции, а не hparam)

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3, # Начинаем с этого, возможно, придется УМЕНЬШИТЬ позже (до 5e-4 или 1e-4), когда внимание начнет сходиться
        weight_decay=1e-6,
        grad_clip_thresh=1.0, # Стандартное значение, помогает предотвратить взрыв градиентов
        # !!! ЗНАЧИТЕЛЬНО УВЕЛИЧИВАЕМ batch_size для вашей GPU !!!
        batch_size=64, # Начните с 64 при fp16_run=True. Контролируйте VRAM, можно пробовать 48, 80, 96...
        mask_padding=True,

        ################################
        # FINE-TUNE #
        ################################
        use_mmi=False,
        drop_frame_rate=0.0,
        use_gaf=False,
        update_gaf_every_n_step=10,
        max_gaf=0.5,
        global_mean_npy='ruslan_global_mean.npy', # Если вы не используете нормализацию на основе global mean/std, этот файл не нужен
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams

