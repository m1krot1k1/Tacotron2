import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

# Измененный импорт: импортируем AttrDict и build_env из env
# Также импортируем функции для чекпоинтов и plot_spectrogram из env
from env import AttrDict, build_env, load_checkpoint, save_checkpoint, scan_checkpoint, plot_spectrogram

# Импорт meldataset и models
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss


torch.backends.cudnn.benchmark = True


def train(rank, a, h):
    """
    Основная функция тренировки HiFi-GAN.
    """
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # --- ИСПРАВЛЕНИЕ: Определяем директорию для сохранения моделей из hparams ---
    # Убедитесь, что 'checkpoint_output_directory' существует в вашем config.json
    output_directory = h.checkpoint_output_directory
    # -----------------------------------------------------------------------

    # --- ИСПРАВЛЕНИЕ: Создаем директорию для сохранения, если она не существует (через build_env) ---
    # build_env теперь просто создает директорию, если ее нет как директории
    if rank == 0:
         print(generator)
         # Удалена старая строка os.makedirs(a.checkpoint_path, exist_ok=True) которая была некорректна
         build_env(output_directory) # <<< ИСПРАВЛЕНО: Передаем директорию сохранения в build_env
         print("Директория для сохранения чекпоинтов: ", output_directory)
    # -----------------------------------------------------------------------


    # --- ИСПРАВЛЕНИЕ: Корректно сканируем чекпоинты из директории сохранения ---
    # scan_checkpoint теперь ожидает директорию для поиска возобновления обучения
    cp_g = scan_checkpoint(output_directory, 'g_')
    cp_do = scan_checkpoint(output_directory, 'do_')
    # -----------------------------------------------------------------------


    steps = 0
    last_epoch = -1

    # Логика загрузки чекпоинта: сначала пробуем загрузить из output_directory для возобновления,
    # если не найдены И указан a.checkpoint_path (из bash для fine-tuning), загружаем его.
    if cp_g is None or cp_do is None:
        state_dict_do = None # Сбрасываем state_dict_do, если нет полных чекпоинтов для возобновления

        # Проверяем, указан ли путь к исходному чекпоинту для fine-tuning
        if a.checkpoint_path is not None and os.path.isfile(a.checkpoint_path):
             print(f"Начинаем fine-tuning. Загрузка генератора из {a.checkpoint_path}")
             state_dict_g = load_checkpoint(a.checkpoint_path, device) # load_checkpoint теперь в env
             if state_dict_g and 'generator' in state_dict_g:
                generator.load_state_dict(state_dict_g['generator'])
             else:
                print(f"Предупреждение: Чекпоинт {a.checkpoint_path} не содержит состояния 'generator'. Начинаем с нуля.")
                # Если не удалось загрузить генератор из указанного чекпоинта, начинаем с нуля
                state_dict_do = None # Сброс на всякий случай
                steps = 0
                last_epoch = -1

             # Дискриминаторы и оптимизаторы обычно не загружаются при warm-start/fine-tuning
             # state_dict_do = None # Уже сброшен выше, если cp_g/cp_do None
             steps = 0 # Начинаем шаги с 0 при fine-tuning
             last_epoch = -1 # Начинаем эпохи с 0 при fine-tuning

        else:
             print("Начинаем обучение с нуля (чекпоинты в директории сохранения не найдены, и initial_checkpoint не указан или не существует).")
             state_dict_do = None # Убедимся, что state_dict_do None при обучении с нуля
             steps = 0
             last_epoch = -1
    else:
        print(f"Возобновление обучения из чекпоинтов в {output_directory}")
        state_dict_g = load_checkpoint(cp_g, device) # load_checkpoint теперь в env
        state_dict_do = load_checkpoint(cp_do, device) # load_checkpoint теперь в env
        if state_dict_g and 'generator' in state_dict_g:
            generator.load_state_dict(state_dict_g['generator'])
        else:
            print(f"Ошибка: Чекпоинт генератора {cp_g} не содержит состояния 'generator'. Не удается возобновить обучение.")
            exit(1) # Выход при невозможности возобновления генератора

        if state_dict_do and 'mpd' in state_dict_do and 'msd' in state_dict_do and 'optim_g' in state_dict_do and 'optim_d' in state_dict_do and 'steps' in state_dict_do and 'epoch' in state_dict_do:
             mpd.load_state_dict(state_dict_do['mpd'])
             msd.load_state_dict(state_dict_do['msd'])
             steps = state_dict_do['steps'] + 1
             last_epoch = state_dict_do['epoch']
             # Загружаем оптимизаторы только если загрузили state_dict_do
             # Если загрузка оптимизаторов происходит после их создания, нужно проверить их состояние
             # Осторожно: загрузка state_dict оптимизатора может потребовать,
             # чтобы параметры модели были уже загружены.
             try:
                 optim_g.load_state_dict(state_dict_do['optim_g'])
                 optim_d.load_state_dict(state_dict_do['optim_d'])
             except RuntimeError as e:
                 print(f"Предупреждение: Не удалось загрузить состояние оптимизаторов. Начинаем обучение с новыми оптимизаторами. Ошибка: {e}")
                 # Сбрасываем state_dict_do, чтобы не использовать старые шаги/эпоху с новыми оптимизаторами
                 state_dict_do = None
                 steps = 0
                 last_epoch = -1
        else:
             print(f"Ошибка: Чекпоинт дискриминаторов/оптимизаторов {cp_do} не полон или поврежден. Не удается возобновить обучение.")
             exit(1) # Выход при невозможности возобновления дискриминаторов/оптимизаторов


    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                 h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    # Перезагрузка state_dict оптимизаторов теперь происходит сразу после загрузки модели state_dict_do

    # Инициализируем шедулеры после загрузки state_dict оптимизаторов
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)


    # get_dataset_filelist ожидает объект с путями файлов
    # Поскольку train.py принимает --input_training_file, --input_validation_file
    # и --input_wavs_dir через argparse (объект a), передаем объект a в get_dataset_filelist.
    # get_dataset_filelist теперь использует a.input_wavs_dir как базовый путь для WAV
    training_filelist, validation_filelist = get_dataset_filelist(a)


    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
                          split=True, shuffle=True, # Параметры split и shuffle установлены
                          n_cache_reuse=h.n_cache_reuse, # Использован n_cache_reuse из hparams
                          device=device, fmax_loss=h.fmax_for_loss, fine_tuning=a.fine_tuning,
                          base_mels_path=a.input_mels_dir) # base_mels_path из argparse (объект a)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    # num_workers и batch_size берутся из hparams (объект h)
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
                              split=False, shuffle=False, # split=False, shuffle=False для валидации
                              n_cache_reuse=0, # Не кэшируем на валидации
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir) # input_mels_dir из argparse (объект a)

        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1, # Размер батча 1 для валидации
                                       pin_memory=True,
                                       drop_last=True)

        # --- ИСПРАВЛЕНИЕ: SummaryWriter использует директорию сохранения ---
        sw = SummaryWriter(os.path.join(output_directory, 'logs')) # <<< ИСПРАВЛЕНО: используем output_directory
        # -------------------------------------------------------------------

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs): # a.training_epochs из argparse
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            # filename теперь третий возвращаемый аргумент из __getitem__
            x, y, filename, y_mel = batch # <<< Получаем filename

            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * h.gen_loss_mel_weight # Использован вес из hparams (предполагается в конфиге)

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g) * h.gen_loss_fm_weight # Использован вес из hparams
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g) * h.gen_loss_fm_weight # Использован вес из hparams
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g) * h.gen_loss_gen_weight # Использован вес из hparams
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g) * h.gen_loss_gen_weight # Использован вес из hparams
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all.item(), mel_error, time.time() - start_b)) # .item() для вывода тензоров

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    # --- ИСПРАВЛЕНИЕ: Путь сохранения формируется на основе output_directory ---
                    checkpoint_path_g = "{}/g_{:08d}".format(output_directory, steps) # <<< ИСПРАВЛЕНО
                    save_checkpoint(checkpoint_path_g, # save_checkpoint теперь в env
                                     {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path_do = "{}/do_{:08d}".format(output_directory, steps) # <<< ИСПРАВЛЕНО
                    save_checkpoint(checkpoint_path_do, # save_checkpoint теперь в env
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})
                    # -------------------------------------------------------------------------


                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all.item(), steps) # .item() для вывода тензоров
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                # Добавлена проверка на rank == 0 перед валидацией
                if rank == 0 and steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        # Добавлена проверка наличия validation_loader
                        if validation_loader is not None:
                            for j, batch in enumerate(validation_loader):
                                # filename теперь третий возвращаемый аргумент из __getitem__
                                x, y, filename_val, y_mel = batch # <<< Получаем filename_val

                                y_g_hat = generator(x.to(device))
                                y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                              h.hop_size, h.win_size,
                                                              h.fmin, h.fmax_for_loss)
                                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                                # <-- Убедитесь, что отступ здесь правильный и одинаковый
                                if j <= 4: # Ограничиваем количество валидационных аудио для логгирования
                                    # plot_spectrogram теперь импортируется из env
                                    # x - это mel spectrogram (tensor) из датасета
                                    sw.add_figure(f'validation/gt_mel_spec_{j}', plot_spectrogram(x[0].cpu().numpy()), steps)

                                    # y - это исходное аудио в тензоре
                                    # sw.add_audio(f'validation/gt_audio_{j}', y[0].squeeze().cpu(), steps, h.sampling_rate) # Убедитесь, что y - это аудио и в нужном формате


                                    # y_g_hat - это сгенерированный аудио тензор [1, 1, T]
                                    # <-- Убедитесь, что отступ здесь правильный и одинаковый, как и у строк выше в этом блоке
                                    sw.add_audio(f'validation/generated_audio_{j}', y_g_hat[0].squeeze().cpu(), steps, h.sampling_rate) # Убедитесь, что формат корректен

                                    # y_hat_spec - это сгенерированный mel spectrogram (tensor)
                                    # plot_spectrogram ожидает numpy массив
                                    y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                                 h.sampling_rate, h.hop_size, h.win_size,
                                                                 h.fmin, h.fmax) # fmax без _loss для валидации графика
                                    sw.add_figure(f'validation/generated_mel_spec_{j}',
                                                  plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)


                            # Добавлена проверка, чтобы избежать деления на ноль, если validation_loader пуст
                            # <-- Этот блок кода должен иметь отступ, соответствующий циклу 'for j, batch in enumerate(validation_loader):'
                            if j >= 0: # Проверяем, что цикл выполнился хотя бы один раз
                                val_err = val_err_tot / (j+1)
                                sw.add_scalar("validation/mel_spec_error", val_err, steps)
                            else:
                                print("Предупреждение: validation_loader пуст.") # Этот print тоже должен быть на том же уровне отступа


                    generator.train() # <-- Этот вызов должен быть на том же уровне отступа, что и 'with torch.no_grad():' и 'if validation_loader is not None:'


            steps += 1

        # Шедулеры шагаются после каждой эпохи
        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    """
    Основная функция запуска тренировочного процесса.
    """
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None, help='Distributed training group name')

    # Пути к данным (файлы списков и директории)
    # input_wavs_dir - ДОЛЖЕН указывать на директорию, где ЛЕЖАТ WAV ФАЙЛЫ
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs', help='Path to input wavs directory (used by MelDataset and get_dataset_filelist to find audio)')
    # input_mels_dir - ДОЛЖЕН указывать на директорию, где ЛЕЖАТ СГЕНЕРИРОВАННЫЕ МЕЛЫ для fine-tuning
    parser.add_argument('--input_mels_dir', default='ft_dataset', help='Path to input mels directory (used by MelDataset in fine-tuning mode)')
    # input_training_file и input_validation_file - пути к ФАЙЛАМ списков WAV/Mel + транскрипция
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt', help='Path to the training file list (.txt or .csv)')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt', help='Path to the validation file list (.txt or .csv)')


    # checkpoint_path - путь к ИСХОДНОМУ чекпоинту для fine-tuning (если есть)
    parser.add_argument('--checkpoint_path', default=None, help='Path to an initial checkpoint to fine-tune from (e.g., a pretrained generator model file)')

    # config - путь к файлу конфигурации
    parser.add_argument('--config', default='', required=True, help='Path to the configuration file (.json)')

    # training_epochs - количество эпох для обучения
    parser.add_argument('--training_epochs', default=3100, type=int, help='Number of training epochs')

    # Интервалы логирования и сохранения (в шагах тренировки)
    parser.add_argument('--stdout_interval', default=5, type=int, help='Interval for logging to stdout (in steps)')
    parser.add_argument('--checkpoint_interval', default=5000, type=int, help='Interval for saving checkpoints (in steps)')
    parser.add_argument('--summary_interval', default=100, type=int, help='Interval for logging summaries to TensorBoard (in steps)')
    parser.add_argument('--validation_interval', default=1000, type=int, help='Interval for running validation (in steps)')

    # fine_tuning флаг (булев флаг, не требует значения true/false)
    parser.add_argument('--fine_tuning', action='store_true', help='Enable fine-tuning mode')

    # Добавлен аргумент для отключения распределенного обучения, если нужно
    parser.add_argument('--disable_distributed', action='store_true', help='Disable distributed training even if multiple GPUs are available')


    # Аргументы, которые, как мы выяснили, ваша версия не обрабатывает из командной строки,
    # но они могут быть в конфиге: batch_size, learning_rate, num_workers, num_gpus, ...
    # output_directory НЕ ДОЛЖЕН БЫТЬ АРГУМЕНТОМ КОМАНДНОЙ СТРОКИ, он должен быть в конфиге!


    a = parser.parse_args()

    # Проверка наличия файла конфига
    if not os.path.isfile(a.config):
        print(f"Ошибка: Файл конфигурации не найден по пути: {a.config}")
        parser.print_help()
        exit(1)

    with open(a.config) as f:
        data = f.read()

    try:
        json_config = json.loads(data)
    except json.JSONDecodeError as e:
        print(f"Ошибка при парсинге файла конфигурации {a.config}: {e}")
        print("Пожалуйста, проверьте синтаксис JSON в файле (особенно кавычки и запятые).")
        exit(1)

    h = AttrDict(json_config)

    # --- ИСПРАВЛЕНИЕ: Проверяем наличие обязательных параметров в конфиге ---
    required_config_params = ['checkpoint_output_directory', 'n_cache_reuse', 'gen_loss_mel_weight', 'gen_loss_fm_weight', 'gen_loss_gen_weight', 'num_workers', 'batch_size'] # Добавлены другие параметры, используемые в train()
    for param in required_config_params:
        if param not in h:
             print(f"Ошибка: Обязательный параметр '{param}' отсутствует в файле конфигурации {a.config}")
             print(f"Пожалуйста, добавьте '{param}' в ваш config.json")
             exit(1)
    # ---------------------------------------------------------------------------------------


    # --- ИСПРАВЛЕНИЕ: Передаем правильный путь к выходной директории в build_env ---
    # build_env теперь создает директорию, указанную в конфиге
    # build_env также проверяет, существует ли путь И является ли он директорией
    build_env(h.checkpoint_output_directory) # <<< ИСПРАВЛЕНО: используем директорию из конфига
    # -------------------------------------------------------------------------------


    torch.manual_seed(h.seed)
    # num_gpus и batch_size должны быть в hparams (config.json)
    if torch.cuda.is_available() and not a.disable_distributed: # Добавлена проверка флага disable_distributed
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        # Предполагаем, что batch_size в конфиге - это batch_size_per_gpu
        # Если batch_size в конфиге - это total_batch_size, то нужно раскомментировать:
        # h.batch_size = int(h.batch_size / h.num_gpus)
        print('Используется GPU. Количество GPU:', h.num_gpus)
        print('Размер батча на GPU :', h.batch_size)

        # Добавлена проверка необходимости инициализации распределенного процесса
        # Это произойдет только если num_gpus > 1 И distributed не отключено
        if h.num_gpus > 1:
             print("Инициализация распределенного обучения...")
             # Проверка dist_config
             if 'dist_config' not in h or 'dist_backend' not in h.dist_config or 'dist_url' not in h.dist_config or 'world_size' not in h.dist_config:
                 print("Ошибка: Параметры 'dist_config' (dist_backend, dist_url, world_size) отсутствуют или неполны в файле конфигурации для распределенного обучения.")
                 exit(1)
             # Здесь сам init_process_group происходит внутри mp.spawn(train, ...)
        else:
             print("Используется только 1 GPU. Распределенное обучение не будет запущено.")

    else:
         h.num_gpus = 0 # Или 1 для CPU
         print('GPU не обнаружено или распределенное обучение отключено. Использование CPU.')
         print('Размер батча :', h.batch_size) # Размер батча для CPU


    # Запуск тренировочного процесса
    if h.num_gpus > 1 and not a.disable_distributed: # Запускаем распределенное обучение только если несколько GPU и не отключено
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h) # Запускаем обычное обучение (1 GPU или CPU)


if __name__ == '__main__':
    main()
