import os
import re
import shutil
import argparse
import sys # Добавлено для sys.exit
import glob # Добавлено для поиска файлов
import random # Добавлено для перемешивания и разделения

def normalize_text(text):
    # Удаляем лишние пробелы и переводы строк
    text = text.strip()
    text = re.sub(r'\s+', ' ', text) # Заменяем множественные пробелы на один

    # Удаляем стандартные символы, которые часто не нужны
    text = text.replace('"', '').replace('...', '.').replace('…', '.')

    # Заменяем символ ё на е, если это необходимо для вашей модели
    # ВАЖНО: Убедитесь, что ваш словарь модели (если используется) или символ_сет
    # в hparams.py содержит либо 'ё', либо вы должны выполнить эту замену.
    # Обычно модели работают лучше с 'е' вместо 'ё'.
    text = text.replace('ё', 'е').replace('Ё', 'Е')

    # Удаляем символы в скобках [] () {}
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)

    # Удаляем другие специальные символы, которые могут быть в выводе Whisper
    # ВАЖНО: Оставьте только те символы, которые ваша модель обучена обрабатывать!
    # Текущий набор оставляет кириллицу, латиницу, пробелы и основные знаки препинания.
    # Проверьте hparams.py вашей модели на полный список символов.
    # Пример расширенного набора: "абвгдежзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:-"
    allowed_chars = "абвгдежзийклмнопрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?;:" # Убедитесь, что этот набор соответствует вашей модели!
    # Удаляем символы, НЕ входящие в разрешенный набор
    text = ''.join(c for c in text if c in allowed_chars)


    # Приводим к нижнему регистру (ВАЖНО: зависит от того, как обучалась модель!)
    # Большинство моделей Tacotron 2 работают с текстом в нижнем регистре.
    text = text.lower()


    # Добавляем точку в конце, если ее нет и текст непустой
    if text and not text[-1] in ['.', '!', '?']:
        text += '.'
    # Удаляем пробелы в начале/конце после всех операций
    text = text.strip()

    return text


def prepare_and_split_metadata(trans_dir, audio_dir, dataset_dir, metadata_file_all, train_file, validation_file, validation_split=0.1):
    # Ищем файлы транскрипций по шаблону *.txt в указанной директории (без рекурсии)
    txt_files_paths = sorted(glob.glob(os.path.join(trans_dir, '*.txt')))
    if not txt_files_paths:
        print(f"[ERROR] В директории {trans_dir} не найдено файлов транскрипций.")
        return False

    total_files = len(txt_files_paths)
    print(f"[INFO] Найдено {total_files} файлов транскрипций для обработки.")

    wavs_dir = os.path.join(dataset_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True) # Убедимся, что директория wavs существует в директории датасета
    processed_entries = [] # Список для хранения всех обработанных записей

    print(f"[INFO] Создание полного файла метаданных: {metadata_file_all}")

    with open(metadata_file_all, "w", encoding="utf-8") as meta_f:
        for i, txt_file_path in enumerate(txt_files_paths, 1):
            file = os.path.basename(txt_file_path)
            base_name = file.replace('.txt', '')
            wav_file = base_name + '.wav'
            input_wav_path = os.path.join(audio_dir, wav_file) # Путь к WAV в processed_audio

            # Проверяем существование соответствующего WAV файла в processed_dir
            if not os.path.exists(input_wav_path):
                print(f"[{i}/{total_files}] ПРОПУСК {file}: соответствующий WAV-файл '{wav_file}' не найден в {audio_dir}")
                continue # Пропускаем файл, если нет соответствующего WAV

            try:
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except Exception as e:
                print(f"[{i}/{total_files}] ПРОПУСК {file}: Ошибка чтения файла транскрипции: {e}")
                continue # Пропускаем файл при ошибке чтения

            text = normalize_text(text)
            if not text:
                print(f"[{i}/{total_files}] ПРОПУСК {file}: пустая или некорректная транскрипция после нормализации")
                continue # Пропускаем, если нормализованный текст пуст

            output_wav_path = os.path.join(wavs_dir, wav_file) # Путь куда копируем WAV в dataset/wavs
            try:
                # Копируем файл в директорию датасета/wavs
                shutil.copy2(input_wav_path, output_wav_path)
                # Добавляем запись в список обработанных
                # Формат: имя_файла.wav|текст_транскрипции|пустое_поле
                processed_entries.append(f"{os.path.basename(output_wav_path)}|{text}|\n")
                # print(f"[{i}/{total_files}] Обработан {file}") # Убрал детальный вывод здесь для краткости
            except Exception as e:
                print(f"[{i}/{total_files}] ПРОПУСК {file}: Ошибка при копировании {wav_file}: {e}")
                continue # Пропускаем файл при ошибке копирования

        # Записываем все обработанные записи в полный файл metadata.csv
        for entry in processed_entries:
            meta_f.write(entry)

    if not processed_entries:
        print(f"[ERROR] Подготовка датасета завершена, но не удалось обработать ни одного файла. Проверьте ошибки.")
        # Удаляем пустые файлы метаданных
        if os.path.exists(metadata_file_all): os.remove(metadata_file_all)
        if os.path.exists(train_file): os.remove(train_file)
        if os.path.exists(validation_file): os.remove(validation_file)
        return False

    # Перемешиваем записи для случайного разделения
    random.shuffle(processed_entries)
    total_processed = len(processed_entries)
    validation_size = int(total_processed * validation_split)
    train_size = total_processed - validation_size

    if train_size == 0 or validation_size == 0:
        print(f"[WARNING] После обработки осталось {total_processed} валидных записей. Недостаточно данных для разделения на обучение ({train_size}) и валидацию ({validation_size}).")
        print("[WARNING] Все данные будут записаны только в training_files.")
        with open(train_file, "w", encoding="utf-8") as train_f:
            for entry in processed_entries:
                train_f.write(entry)
        # Создаем пустой файл валидации, чтобы hparams не выдал ошибку
        open(validation_file, 'w').close()
        print(f"[SUCCESS] Все {total_processed} записей записаны в {os.path.basename(train_file)}.")
        print(f"[INFO] Создан пустой файл валидации: {os.path.basename(validation_file)}.")
        print(f"[INFO] WAV-файлы скопированы в {os.path.join(dataset_dir, 'wavs')}")
        return True # Считаем успешным, даже если нет разделения

    # Разделяем на обучающий и валидационный наборы
    train_entries = processed_entries[:train_size]
    validation_entries = processed_entries[train_size:]

    # Записываем в train.csv
    print(f"[INFO] Запись {len(train_entries)} записей в {os.path.basename(train_file)}...")
    with open(train_file, "w", encoding="utf-8") as train_f:
        for entry in train_entries:
            train_f.write(entry)
    print(f"[SUCCESS] Файл {os.path.basename(train_file)} создан.")

    # Записываем в validation.csv
    print(f"[INFO] Запись {len(validation_entries)} записей в {os.path.basename(validation_file)}...")
    with open(validation_file, "w", encoding="utf-8") as validation_f:
        for entry in validation_entries:
            validation_f.write(entry)
    print(f"[SUCCESS] Файл {os.path.basename(validation_file)} создан.")


    print(f"[SUCCESS] Подготовка датасета и разделение завершены.")
    print(f"[INFO] Полный файл метаданных (не используется напрямую для обучения): {metadata_file_all}")
    print(f"[INFO] Файл обучающих данных: {train_file}")
    print(f"[INFO] Файл валидационных данных: {validation_file}")
    print(f"[INFO] WAV-файлы скопированы в: {wavs_dir}")

    # Добавляем отладочный вывод содержимого train.csv
    print("[INFO] Предварительный просмотр train.csv (первые 5 строк):")
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for j, line in enumerate(lines):
                if j < 5: # Выводим только первые 5 строк
                    print(line.strip())
                else:
                    print(f"... еще {len(lines) - 5} строк")
                    break
            if not lines:
                 print("(Файл train.csv пуст)")
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать train.csv для просмотра: {e}")

    # Добавляем отладочный вывод содержимого validation.csv
    print("[INFO] Предварительный просмотр validation.csv (первые 5 строк):")
    try:
        with open(validation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for j, line in enumerate(lines):
                if j < 5: # Выводим только первые 5 строк
                    print(line.strip())
                else:
                    print(f"... еще {len(lines) - 5} строк")
                    break
            if not lines:
                 print("(Файл validation.csv пуст)")
    except Exception as e:
        print(f"[ERROR] Не удалось прочитать validation.csv для просмотра: {e}")


    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and split metadata for Tacotron 2 dataset")
    parser.add_argument("--trans_dir", type=str, required=True, help="Directory with transcription files")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory with processed WAV files")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory to save dataset")
    parser.add_argument("--metadata_file_all", type=str, required=True, help="Path to combined metadata file (intermediate)")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training metadata file")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation metadata file")
    parser.add_argument("--validation_split", type=float, default=0.1, help="Fraction of data to use for validation")
    args = parser.parse_args()

    if not prepare_and_split_metadata(args.trans_dir, args.audio_dir, args.dataset_dir, args.metadata_file_all, args.train_file, args.validation_file, args.validation_split):
        sys.exit(1) # Выходим с ошибкой, если датасет не создан
