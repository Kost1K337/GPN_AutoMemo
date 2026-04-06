"""
Генерация eval-датасета для ASR бенчмарка.
Использует Silero TTS — полностью офлайн, без GPU, российская разработка.

Установка:
  pip install torch torchaudio omegaconf numpy --index-url https://download.pytorch.org/whl/cpu

Запуск:
  python generate_eval_dataset_silero.py
"""

import json
import os
import re
import random
import time
import torch

# ============================================================
# Фонетическое раскрытие аббревиатур для TTS
# ============================================================
LETTER_PHONETICS = {
    'А': 'а', 'Б': 'бэ', 'В': 'вэ', 'Г': 'гэ', 'Д': 'дэ',
    'Е': 'е', 'Ж': 'жэ', 'З': 'зэ', 'И': 'и', 'К': 'ка',
    'Л': 'эль', 'М': 'эм', 'Н': 'эн', 'О': 'о', 'П': 'пэ',
    'Р': 'эр', 'С': 'эс', 'Т': 'тэ', 'У': 'у', 'Ф': 'эф',
    'Х': 'ха', 'Ц': 'цэ', 'Ч': 'чэ', 'Ш': 'ша', 'Щ': 'ща',
    'Э': 'э', 'Ю': 'ю', 'Я': 'я',
}

def expand_abbreviations(text: str) -> str:
    """Заменяет аббревиатуры (2+ заглавных кириллических букв подряд)
    на побуквенное произношение для TTS.
    'УПН' → 'у пэ эн', 'ГРП' → 'гэ эр пэ'
    """
    def replace_match(m):
        abbr = m.group(0)
        return ' '.join(LETTER_PHONETICS.get(ch, ch) for ch in abbr)

    return re.sub(r'[А-ЯЁ]{2,}', replace_match, text)

# ============================================================

# Загружаем модель Silero TTS (при первом запуске скачает ~100 МБ)
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v4_ru'
)

# Доступные русские голоса в v4_ru
RU_SPEAKERS = ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']

SAMPLE_RATE = 48000

def main():
    with open("eval_sentences.txt", "r", encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]

    print(f"📋 Загружено {len(sentences)} предложений")

    total_start = time.time()
    os.makedirs("eval_dataset/audio", exist_ok=True)
    os.makedirs("eval_dataset/text", exist_ok=True)

    references = {}

    for i, sentence in enumerate(sentences):
        speaker = random.choice(RU_SPEAKERS[:5])
        audio_path = f"eval_dataset/audio/audio_{i:03d}.wav"
        text_path = f"eval_dataset/text/text_{i:03d}.txt"

        # TTS получает фонетический текст, reference — оригинал
        tts_text = expand_abbreviations(sentence)

        t0 = time.time()

        audio = model.save_wav(
            text=tts_text,
            speaker=speaker,
            sample_rate=SAMPLE_RATE,
            audio_path=audio_path
        )

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(sentence)

        elapsed = time.time() - t0
        print(f"  [{i+1:02d}/{len(sentences)}] {speaker}: {sentence[:60]}... ({elapsed:.2f}с)")

        references[f"audio_{i:03d}"] = {
            "text": sentence,
            "speaker": speaker
        }

    with open("eval_dataset/references.json", "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    total_time = time.time() - total_start
    minutes, seconds = divmod(total_time, 60)
    print(f"\n✅ Готово! {len(references)} пар в eval_dataset/")
    print(f"⏱️  Общее время: {int(minutes)} мин {seconds:.1f} сек ({total_time/len(references):.2f} сек/файл)")


if __name__ == "__main__":
    main()
