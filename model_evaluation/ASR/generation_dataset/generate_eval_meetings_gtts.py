"""
Генерация длинных (до ~5 минут) аудиозаписей совещаний для ASR бенчмарка.
Использует gTTS (Google Translate TTS).

Установка:
  pip install gTTS

Запуск:
  python generate_eval_meetings_gtts.py
"""

import json
import os
import time
from gtts import gTTS

def main():
    meetings_file = "eval_meetings.txt"
    if not os.path.exists(meetings_file):
        print(f"❌ Файл {meetings_file} не найден!")
        return

    with open(meetings_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Разделяем файл по меткам ===MEETING_X===
    # Первый элемент будет пустым (до первой метки), поэтому пропускаем его [1:]
    meetings = content.split("===MEETING_")[1:]
    
    print(f"📋 Найдено {len(meetings)} совещаний")

    output_dir = "eval_dataset_meetings_gtts"
    os.makedirs(f"{output_dir}/audio", exist_ok=True)
    os.makedirs(f"{output_dir}/text", exist_ok=True)

    references = {}
    total_start = time.time()

    for meeting_raw in meetings:
        # Структура: "1===\nТекст..."
        try:
            header, text = meeting_raw.split("===\n", 1)
            meeting_num = int(header.strip())
            text = text.strip()
        except ValueError:
            print("⚠️ Ошибка парсинга метки совещания, пропускаем.")
            continue

        if not text:
            continue

        audio_path = f"{output_dir}/audio/meeting_{meeting_num:02d}.mp3"
        text_path = f"{output_dir}/text/meeting_{meeting_num:02d}.txt"

        t0 = time.time()
        print(f"  ⏳ Генерирую meeting_{meeting_num:02d}.mp3 ({len(text)} символов)...", end="", flush=True)

        try:
            tts = gTTS(text=text, lang='ru')
            tts.save(audio_path)
            
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            elapsed = time.time() - t0
            print(f" готово за {elapsed:.2f}с")

            references[f"meeting_{meeting_num:02d}"] = {
                "text": text,
                "tts": "gTTS"
            }
        except Exception as e:
            print(f"\n  ❌ Ошибка генерации: {e}")

    with open(f"{output_dir}/references.json", "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    total_time = time.time() - total_start
    minutes, seconds = divmod(total_time, 60)
    print(f"\n✅ Готово! {len(references)} записей в {output_dir}/")
    if references:
      print(f"⏱️  Общее время: {int(minutes)} мин {seconds:.1f} сек ({total_time/len(references):.2f} сек/файл)")

if __name__ == "__main__":
    main()
