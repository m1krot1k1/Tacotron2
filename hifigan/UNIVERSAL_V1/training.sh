#!/bin/bash

# Установка флага для прерывания скрипта при ошибках
set -e
# set -u # Закомментировано, так как переменные могут быть пустыми в определенных сценариях или могут быть инициализированы позже

# ==============================================================================
# КОНФИГУРИРУЕМЫЕ ПАРАМЕТРЫ СКРИПТА
# ==============================================================================

# Корень вашего проекта (директория, содержащая Tacotron2, audio, dataset и т.д.)
# УКАЖИТЕ ВАШ АКТУАЛЬНЫЙ ПУТЬ:
PROJECT_ROOT="/home/m1krot1k/Рабочий стол/tacotron"

# Порт для TensorBoard, отслеживающего логи Tacotron 2 (из logdir)
TACOTRON_TENSORBOARD_PORT=9000

# Порт для TensorBoard, отслеживающего логи HiFi-GAN (из hifigan/logs)
HIFIGAN_TENSORBOARD_PORT=9001

# ==============================================================================
# АВТОМАТИЧЕСКИ ОПРЕДЕЛЯЕМЫЕ ПУТИ (НЕ РЕКОМЕНДУЕТСЯ ИЗМЕНЯТЬ)
# ==============================================================================

# Директория репозитория Tacotron2
REPO_ROOT="$PROJECT_ROOT/Tacotron2"

# Директория скриптов HiFi-GAN
HIFIGAN_DIR="$REPO_ROOT/hifigan"

# Директория виртуального окружения
VENV_PATH="$PROJECT_ROOT/tacotron2_venv_py37"

# Директория логов Tacotron 2
TACOTRON_LOG_DIR="$REPO_ROOT/logdir"

# Директория логов HiFi-GAN
HIFIGAN_LOG_DIR="$HIFIGAN_DIR/logs"

# Путь к файлу Streamlit демо
DEMO_PY_PATH="$REPO_ROOT/demo.py"

# Путь к конфигу HiFi-GAN относительно директории HiFi-GAN
# Убедитесь, что этот файл существует по этому пути внутри $HIFIGAN_DIR
HIFIGAN_CONFIG_RELATIVE_PATH="UNIVERSAL_V1/config.json"

# Путь к предобученному чекпоинту HiFi-GAN относительно директории HiFi-GAN
# Убедитесь, что этот файл существует по этому пути внутри $HIFIGAN_DIR
HIFIGAN_CHECKPOINT_RELATIVE_PATH="UNIVERSAL_V1/g_02500000"

# Путь к директории сгенерированных мел-спектрограмм
GENERATED_MELS_DIR="$REPO_ROOT/data/wavs" # Сгенерированные мелсы сохраняются в корне проекта, в hifigan/training_mels

# Путь к файлам тренировочного и валидационного датасетов (они копируются в $REPO_ROOT/data на шаге 6 общего скрипта)
HIFIGAN_TRAIN_DATA_FILE="$REPO_ROOT/data/train.csv"
HIFIGAN_VALIDATION_DATA_FILE="$REPO_ROOT/data/validation.csv"

# Директория для сохранения дообученных чекпоинтов HiFi-GAN (относительно директории HiFi-GAN)
# ВНИМАНИЕ: В ВАШЕЙ ВЕРСИИ HIFIGAN/TRAIN.PY ЭТОТ ПУТЬ ДОЛЖЕН БЫТЬ ПРОПИСАН В КОНФИГЕ И ПРАВИЛЬНО ИСПОЛЬЗОВАН В train.py!
HIFIGAN_OUTPUT_DIR_RELATIVE="finetuned_models" # Ожидаемое место сохранения моделей (должно быть в конфиге и использоваться в train.py)

# Путь к директории с WAV-файлами датасета относительно директории HiFi-GAN ($HIFIGAN_DIR)
# Ваша директория wavs находится в $REPO_ROOT/data/wavs,
# а HIFIGAN_DIR находится в $REPO_ROOT/hifigan.
# Относительный путь от hifigan/ до data/wavs/ это ../data/wavs/
HIFIGAN_WAVS_DIR_RELATIVE="../data/wavs" # <<< ДОБАВЛЕН/УТОЧНЕН ПУТЬ К WAV

# Параметры дообучения HiFi-GAN (передаются как аргументы командной строки, если скрипт их поддерживает)
# ВНИМАНИЕ: В ВАШЕЙ ВЕРСИИ HIFIGAN/TRAIN.PY НЕКОТОРЫЕ ИЗ ЭТИХ АРГУМЕНТОВ (BATCH_SIZE, LEARNING_RATE, OUTPUT_DIRECTORY) НЕ РАСПОЗНАЮТСЯ
# И ДОЛЖНЫ БЫТЬ ПРОПИСАНЫ В КОНФИГЕ ($HIFIGAN_CONFIG_RELATIVE_PATH).
HIFIGAN_TRAINING_EPOCHS=1000 # Количество эпох для дообучения (используется как аргумент)


# Цвета для вывода в консоль
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Вспомогательные функции для вывода цветных сообщений
print_info()      { echo -e "${BLUE}[INFO] $1${NC}"; }
print_success()   { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
print_error()     { echo -e "${RED}[ERROR] $1${NC}"; }

# ==============================================================================
# ФУНКЦИЯ СОЗДАНИЯ НЕОБХОДИМЫХ ДИРЕКТОРИЙ (НА УРОВНЕ BASH)
# ==============================================================================

prepare_directories() {
    print_info "Создание необходимых директорий для HiFi-GAN..."
    # Создаем директории, которые могут потребоваться HiFi-GAN и сопутствующим файлам
    mkdir -p "$HIFIGAN_DIR"
    mkdir -p "$HIFIGAN_LOG_DIR"
    mkdir -p "$GENERATED_MELS_DIR" # Директория для мелсов, сгенерированных Tacotron 2 (в корне проекта)
    mkdir -p "$REPO_ROOT/data" # Директория для файлов датасета (train.csv, validation.csv) внутри репозитория
    mkdir -p "$REPO_ROOT/data/wavs" # Директория для WAV файлов датасета (используется при подготовке датасета)
    # Создаем директорию для сохранения дообученных моделей HiFi-GAN (относительно директории HIFIGAN_DIR)
    # Это директория, куда bash ожидает сохранения. ВАША ВЕРСИЯ HIFIGAN/TRAIN.PY МОЖЕТ ТРЕБОВАТЬ УКАЗАНИЯ ЭТОГО ПУТИ В КОНФИГЕ И ЕГО КОРРЕКТНОГО ИСПОЛЬЗОВАНИЯ!
    mkdir -p "$HIFIGAN_DIR/$HIFIGAN_OUTPUT_DIR_RELATIVE"
    print_success "Необходимые директории созданы или уже существуют."
}


# ==============================================================================
# ВЫПОЛНЕНИЕ КОМАНД ЗАПУСКА HIFIGAN И МОНИТОРИНГА
# ==============================================================================

print_info "Начало запуска обучения HiFi-GAN с фоновыми процессами..."

# Создаем необходимые директории на уровне bash
prepare_directories

# Переходим в директорию HiFi-GAN
print_info "Переход в директорию HiFi-GAN: $HIFIGAN_DIR"
cd "$HIFIGAN_DIR" || { print_error "Не удалось перейти в директорию $HIFIGAN_DIR"; exit 1; }
print_success "Текущая директория: $(pwd)"

# Активируем виртуальное окружение
print_info "Активация виртуального окружения: $VENV_PATH"
source "$VENV_PATH/bin/activate" || { print_error "Не удалось активировать виртуальное окружение"; exit 1; }
print_success "Виртуальное окружение активировано: $(which python)"

# Добавляем корень репозитория в PYTHONPATH, чтобы Python мог найти модули из корня (например, utils, которые вы предоставили)
print_info "Добавление $REPO_ROOT в PYTHONPATH..."
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT"
print_success "PYTHONPATH установлен."


# Запуск TensorBoard для логов Tacotron 2 (на порту $TACOTRON_TENSORBOARD_PORT)
# Предполагается, что логи Tacotron 2 уже существуют после его обучения.
print_info "Запуск TensorBoard для логов Tacotron 2 (логи обучения Tacotron 2) на порту $TACOTRON_TENSORBOARD_PORT..."
# Завершаем предыдущие экземпляры на этом порту, если они есть
pkill -f "tensorboard --logdir=$TACOTRON_LOG_DIR --host=0.0.0.0 --port=$TACOTRON_TENSORBOARD_PORT" 2>/dev/null || true
# Запускаем в фоне
tensorboard --logdir="$TACOTRON_LOG_DIR" --host=0.0.0.0 --port="$TACOTRON_TENSORBOARD_PORT" &
TACOTRON_TENSORBOARD_PID=$!
print_success "TensorBoard для Tacotron 2 запущен в фоне с PID $TACOTRON_TENSORBOARD_PID на http://0.0.0.0:$TACOTRON_TENSORBOARD_PORT"

# Запуск Streamlit демо (если demo.py существует в корне репозитория Tacotron2)
print_info "Запуск Streamlit демо (если файл demo.py присутствует)..."
if [ -f "$DEMO_PY_PATH" ]; then
  # Переходим в корневую директорию репозитория для запуска Streamlit
  # Сохраняем текущую директорию, чтобы вернуться
  CURRENT_DIR=$(pwd)
  cd "$REPO_ROOT" || { print_error "Не удалось перейти в директорию репозитория для Streamlit."; exit 1; }
  # Завершаем предыдущие экземпляры, если они есть (используем более точный паттерн поиска)
  pkill -f "streamlit run $DEMO_PY_PATH" 2>/dev/null || true
  # Запускаем в фоне
  streamlit run "$DEMO_PY_PATH" &
  STREAMLIT_PID=$!
  print_success "Streamlit демо запущен в фоне с PID $STREAMLIT_PID (http://localhost:8501 по умолчанию)."
  # Возвращаемся в директорию HiFi-GAN
  cd "$CURRENT_DIR" || { print_error "Не удалось вернуться в исходную директорию."; exit 1; }
else
  print_info "Файл demo.py не найден по пути $DEMO_PY_PATH. Пропуск запуска Streamlit."
fi

# Запуск TensorBoard для логов HiFi-GAN (на порту $HIFIGAN_TENSORBOARD_PORT)
print_info "Запуск TensorBoard для логов HiFi-GAN (логи обучения HiFi-GAN) на порту $HIFIGAN_TENSORBOARD_PORT..."
# Завершаем предыдущие экземпляры на этом порту, если они есть
pkill -f "tensorboard --logdir=$HIFIGAN_LOG_DIR --host=0.0.0.0 --port=$HIFIGAN_TENSORBOARD_PORT" 2>/dev/null || true
# Запускаем в фоне
tensorboard --logdir="$HIFIGAN_LOG_DIR" --host=0.0.0.0 --port="$HIFIGAN_TENSORBOARD_PORT" &
HIFIGAN_TENSORBOARD_PID=$!
print_success "TensorBoard для HiFi-GAN запущен в фоне с PID $HIFIGAN_TENSORBOARD_PID на http://0.0.0.0:$HIFIGAN_TENSORBOARD_PORT"

# Небольшая пауза для запуска фоновых процессов
sleep 3

print_info "Запуск дообучения HiFi-GAN..."

# ==============================================================================
# ЗАПУСК ОСНОВНОГО ПРОЦЕССА ОБУЧЕНИЯ HIFIGAN
# ВНИМАНИЕ: ВАША ВЕРСИЯ HIFIGAN/TRAIN.PY ИМЕЕТ ОШИБКУ В ОБРАБОТКЕ ПУТЕЙ СОХРАНЕНИЯ!
# ОШИБКА "FileExistsError: [Errno 17] File exists: '.../UNIVERSAL_V1/g_02500000'"
# МОЖЕТ ВОЗНИКНУТЬ ВНУТРИ PYTHON СКРИПТА, ЕСЛИ ВЫ НЕ ВНЕСЛИ ПРАВИЛЬНЫЕ ИЗМЕНЕНИЯ В train.py И env.py.
# ЭТА ОШИБКА НЕ МОЖЕТ БЫТЬ ИСПРАВЛЕНА ЭТИМ БАШ-СКРИПТОМ.
# ВАМ НЕОБХОДИМО БЫЛО ОТРЕДАКТИРОВАТЬ ВАШИ PYTHON ФАЙЛЫ train.py И env.py (как я показывал),
# А ТАКЖЕ ДОБАВИТЬ ПАРАМЕТР В config.json.
# ==============================================================================

# Выполняем команду обучения HiFi-GAN, находясь в директории hifigan
# Аргументы batch_size, learning_rate, output_directory ожидаются в файле конфига (.json)
# Мы передаем аргументы, которые train.py, как мы выяснили, ожидает через командную строку
python train.py \
    --config "$HIFIGAN_CONFIG_RELATIVE_PATH" \
    --input_mels_dir "$GENERATED_MELS_DIR" \
    --input_training_file "$HIFIGAN_TRAIN_DATA_FILE" \
    --input_validation_file "$HIFIGAN_VALIDATION_DATA_FILE" \
    --input_wavs_dir "$HIFIGAN_WAVS_DIR_RELATIVE" \
    --checkpoint_path "$HIFIGAN_CHECKPOINT_RELATIVE_PATH" \
    --fine_tuning \
    --training_epochs 1000

# Сохраняем PIDы фоновых процессов в файл, чтобы их можно было легко остановить позже
# Проверяем, были ли запущены фоновые процессы перед сохранением PID
if [ -n "$TACOTRON_TENSORBOARD_PID" ]; then echo "$TACOTRON_TENSORBOARD_PID" > hifigan_bg_pids.txt; else echo "" > hifigan_bg_pids.txt; fi
if [ -n "$HIFIGAN_TENSORBOARD_PID" ]; then echo "$HIFIGAN_TENSORBOARD_PID" >> hifigan_bg_pids.txt; fi
if [ -n "$STREAMLIT_PID" ]; then echo "$STREAMLIT_PID" >> hifigan_bg_pids.txt; fi


print_success "Процесс дообучения HiFi-GAN завершен (или завершился с ошибкой)."

# После завершения обучения HiFi-GAN, вы можете остановить фоновые процессы.
# Для остановки вручную:
# Откройте другой терминал, перейдите в директорию HiFi-GAN ($HIFIGAN_DIR)
# Прочитайте PIDы из файла: cat hifigan_bg_pids.txt
# Остановите каждый процесс командой: kill <PID>
# Или просто выполните команды pkill, которые мы использовали для запуска.
# Например:
# pkill -f "tensorboard --logdir=$TACOTRON_LOG_DIR --host=0.0.0.0 --port=$TACOTRON_TENSORBOARD_PORT"
# pkill -f "tensorboard --logdir=$HIFIGAN_LOG_DIR --host=0.0.0.0 --port=$HIFIGAN_TENSORBOARD_PORT"
# pkill -f "streamlit run $DEMO_PY_PATH"


# Деактивируем виртуальное окружение при завершении скрипта
print_info "Деактивация виртуального окружения..."
deactivate 2>/dev/null || true
print_success "Виртуальное окружение деактивировано."

print_info "Фоновые процессы TensorBoard (порты $TACOTRON_TENSORBOARD_PORT, $HIFIGAN_TENSORBOARD_PORT) и Streamlit (если запущен) продолжают работать."
print_info "Для их остановки используйте команду kill с соответствующими PID или команды pkill."
print_info "PIDы сохранены в файле $HIFIGAN_DIR/hifigan_bg_pids.txt"
