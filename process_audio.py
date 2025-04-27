import os
import argparse
from pydub import AudioSegment
import sys # Добавлено для sys.exit
import glob # Добавлено для поиска файлов

def process_and_split_audio_files(audio_dir, processed_dir, chunk_length_sec=10):
    """
    Обрабатывает MP3-файлы, конвертирует их в WAV, нарезает длинные на чанки.
    """
    os.makedirs(processed_dir, exist_ok=True)
    # Ищем файлы по шаблону *.mp3 в указанной директории (без рекурсии)
    mp3_files_paths = sorted(glob.glob(os.path.join(audio_dir, '*.mp3')))
    if not mp3_files_paths:
        print(f"[ERROR] В директории {audio_dir} не найдено MP3-файлов.")
        return False
    total_chunks = 0
    processed_files_count = 0

    for input_path in mp3_files_paths:
        file = os.path.basename(input_path)
        base_name = os.path.splitext(file)[0]
        print(f"[INFO] Обработка файла: {file}")
        try:
            # Добавлена проверка, что файл существует (хотя glob.glob должен это гарантировать)
            if not os.path.exists(input_path):
                print(f"[WARNING] Файл {file} не найден. Пропуск.")
                continue

            audio = AudioSegment.from_file(input_path)
            # Убедимся в частоте (22050 Гц) и каналах (1, моно) - стандарт для Tacotron 2
            audio = audio.set_frame_rate(22050).set_channels(1)
            duration_ms = len(audio)
            chunk_length_ms = chunk_length_sec * 1000

            if duration_ms == 0:
                print(f"[WARNING] Файл {file} имеет нулевую длительность. Пропуск.")
                continue

            if duration_ms <= chunk_length_ms:
                output_filename = f"{base_name}.wav"
                output_path = os.path.join(processed_dir, output_filename)
                audio.export(output_path, format="wav")
                print(f"[INFO] Сохранен файл: {output_filename} (длительность: {duration_ms/1000:.2f} сек)")
                total_chunks += 1
                processed_files_count += 1

            else:
                num_chunks = (duration_ms + chunk_length_ms - 1) // chunk_length_ms
                print(f"[INFO] Разделение на {num_chunks} фрагментов...")
                chunk_count_for_file = 0
                for i in range(num_chunks):
                    start = i * chunk_length_ms
                    end = min((i + 1) * chunk_length_ms, duration_ms)
                    chunk = audio[start:end]
                    chunk_filename = f"{base_name}_chunk_{i+1:03d}.wav"
                    chunk_path = os.path.join(processed_dir, chunk_filename)
                    chunk.export(chunk_path, format="wav")
                    # print(f"[INFO] Создан файл: {chunk_filename} (длительность: {(end-start)/1000:.2f} сек)") # Отладочный вывод, можно убрать
                    total_chunks += 1
                    chunk_count_for_file += 1


                if chunk_count_for_file > 0:
                     processed_files_count += 1 # Считаем исходный файл обработанным, если из него создан хотя бы 1 чанк


        except Exception as e:
            print(f"[ERROR] Ошибка при обработке файла {file}: {e}")
            # Продолжаем работу, даже если один файл вызвал ошибку


    if total_chunks > 0:
        print(f"[SUCCESS] Обработка завершена. Обработано исходных файлов: {processed_files_count}. Создано {total_chunks} WAV-файлов.")
        # ИСПРАВЛЕНИЕ: Используем аргумент processed_dir вместо неопределенной переменной
        print(f"[INFO] Файлы сохранены в директории: {processed_dir}") # <-- ИСПРАВЛЕНО
        return True
    else:
        print(f"[WARNING] Обработка завершена, но не создано ни одного WAV-файла. Проверьте логи ошибок.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and split audio files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with MP3 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed WAV files")
    parser.add_argument("--chunk_length", type=int, default=10, help="Length of audio chunks in seconds")
    args = parser.parse_args()

    # Передаем аргумент output_dir в функцию process_and_split_audio_files
    if not process_and_split_audio_files(args.input_dir, args.output_dir, args.chunk_length): # <-- Передаем args.output_dir
         sys.exit(1) # Выходим с ошибкой, если не создано ни одного файла
