"""
Генерация eval-датасета для ASR бенчмарка.
Использует edge-tts (Microsoft Edge TTS) — работает на любом Python, без GPU.

Установка:
  pip install edge-tts

Запуск:
  python generate_eval_dataset_edge_tts.py
"""

import asyncio
import json
import os
import random

import edge_tts

# Русские голоса Microsoft Edge
RU_VOICES = [
    "ru-RU-DmitryNeural",    # мужской
    "ru-RU-SvetlanaNeural",  # женский
]

async def generate_dataset():
    # Читаем предложения
    with open("eval_sentences.txt", "r", encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]

    print(f"📋 Загружено {len(sentences)} предложений")

    # Создаём папки
    os.makedirs("eval_dataset/audio", exist_ok=True)
    os.makedirs("eval_dataset/text", exist_ok=True)

    references = {}

    for i, sentence in enumerate(sentences):
        voice = random.choice(RU_VOICES)
        audio_path = f"eval_dataset/audio/audio_{i:03d}.mp3"
        text_path = f"eval_dataset/text/text_{i:03d}.txt"

        print(f"  [{i+1:02d}/{len(sentences)}] {voice}: {sentence[:60]}...")

        communicate = edge_tts.Communicate(sentence, voice)
        await communicate.save(audio_path)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(sentence)

        references[f"audio_{i:03d}"] = {
            "text": sentence,
            "voice": voice
        }

    # Сохраняем референсы
    with open("eval_dataset/references.json", "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Готово! {len(references)} пар сгенерировано в eval_dataset/")


if __name__ == "__main__":
    asyncio.run(generate_dataset())
