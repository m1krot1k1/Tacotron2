import os
import sys
import whisper
import argparse
import warnings # Добавлено для фильтрации предупреждений
import glob # Добавлено для поиска файлов
import torch # Добавлено для проверки CUDA

def transcribe_files(input_dir, output_dir, model_name="base"):
    """
    Транскрибирует WAV-файлы с помощью Whisper.
    Версия, совместимая с Python 3.7 (без использования walrus operator).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Отключаем некоторые предупреждения, которые могут возникать из-за версий библиотек
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)


    # Загрузка модели Whisper
    print(f"[INFO] Загрузка модели Whisper {model_name}...")
    try:
        # Проверяем наличие GPU и используем fp16, если доступно
        use_gpu = torch.cuda.is_available()
        print(f"[INFO] Доступно GPU: {use_gpu}")
        device = "cuda" if use_gpu else "cpu"
        model = whisper.load_model(model_name, device=device)
        print(f"[INFO] Модель загружена на устройство: {device}")
    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке модели Whisper: {e}")
        print("[INFO] Убедитесь, что у вас достаточно места на диске и стабильное интернет-соединение для загрузки модели.")
        print("[INFO] Если используете GPU, проверьте совместимость драйверов и CUDA с PyTorch/TensorFlow и установку whisper с поддержкой GPU.")
        return False

    # Получение списка WAV-файлов
    wav_files_paths = sorted(glob.glob(os.path.join(input_dir, '*.wav'))) # Ищем файлы по шаблону *.wav (без рекурсии)
    if not wav_files_paths:
        print(f"[WARNING] В директории {input_dir} не найдено WAV-файлов для транскрипции. Пропуск.")
        return True # Считаем успешным, если нет файлов для обработки
    total_files = len(wav_files_paths)


    print(f"[INFO] Найдено {total_files} WAV-файлов для транскрипции.")


    # Транскрипция каждого файла
    transcribed_count = 0
    for i, input_path in enumerate(wav_files_paths, 1):
        file = os.path.basename(input_path)
        output_path = os.path.join(output_dir, file.replace('.wav', '.txt'))

        print(f"[{i}/{total_files}] Транскрипция {file}...")


        try:
            # Выполнение транскрипции
            # fp16=use_gpu включает половинную точность, если доступно GPU
            result = model.transcribe(input_path, language="ru", fp16=use_gpu)
            transcription_text = result["text"].strip()

            if not transcription_text:
                 print(f"  [WARNING] Транскрипция для {file} пуста. Пропуск файла.")
                 continue

            # Сохранение транскрипции в файл
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription_text)

            # print(f"  [INFO] Транскрипция сохранена в {output_path}") # Убрал для краткости
            transcribed_count += 1


        except Exception as e:
            print(f"  [ERROR] Ошибка при транскрипции {file}: {e}")
            # Продолжаем работу, даже если один файл вызвал ошибку

    if transcribed_count > 0:
        print(f"[SUCCESS] Транскрипция всех файлов завершена! Успешно транскрибировано {transcribed_count}/{total_files}.")
        return True
    else:
        print(f"[ERROR] Транскрипция завершена, но не создано ни одного файла транскрипции из {total_files} WAV. Проверьте ошибки.")
        return False # Считаем неуспешным, если нет транскрибированных файлов


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe WAV files using Whisper")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with WAV files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save transcriptions")
    parser.add_argument("--model", type=str, default="large", help="Whisper model to use (tiny, base, small, medium, large)")

    args = parser.parse_args()

    if not transcribe_files(args.input_dir, args.output_dir, args.model):
        sys.exit(1) # Выходим с ошибкой, если транскрипция неуспешна
