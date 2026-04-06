"""
Генерация eval-датасета для ASR бенчмарка.
Использует gTTS (Google Translate TTS) — нужен интернет, без GPU.

Установка:
  pip install gTTS

Запуск:
  python generate_eval_dataset_gtts.py
"""

import json
import os
import time
from gtts import gTTS


def main():
    with open("eval_sentences.txt", "r", encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]

    print(f"📋 Загружено {len(sentences)} предложений")

    total_start = time.time()
    os.makedirs("eval_dataset_gtts/audio", exist_ok=True)
    os.makedirs("eval_dataset_gtts/text", exist_ok=True)

    references = {}

    for i, sentence in enumerate(sentences):
        audio_path = f"eval_dataset_gtts/audio/audio_{i:03d}.mp3"
        text_path = f"eval_dataset_gtts/text/text_{i:03d}.txt"

        t0 = time.time()

        tts = gTTS(text=sentence, lang='ru')
        tts.save(audio_path)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(sentence)

        elapsed = time.time() - t0
        print(f"  [{i+1:02d}/{len(sentences)}] {sentence[:60]}... ({elapsed:.2f}с)")

        references[f"audio_{i:03d}"] = {
            "text": sentence,
            "tts": "gTTS"
        }

    with open("eval_dataset_gtts/references.json", "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    total_time = time.time() - total_start
    minutes, seconds = divmod(total_time, 60)
    print(f"\n✅ Готово! {len(references)} пар в eval_dataset_gtts/")
    print(f"⏱️  Общее время: {int(minutes)} мин {seconds:.1f} сек ({total_time/len(references):.2f} сек/файл)")


if __name__ == "__main__":
    main()
