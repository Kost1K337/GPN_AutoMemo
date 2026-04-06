# ====== Ячейка 1: Установка ======
#!pip install TTS

# ====== Ячейка 2: Загрузка файла ======
# Загрузи eval_sentences.txt через Files → Upload в Colab
# или:
from google.colab import files
uploaded = files.upload()  # выбери eval_sentences.txt

# ====== Ячейка 3: Генерация датасета ======
import os, json, random
from tqdm import tqdm
from TTS.api import TTS
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Читаем предложения
with open("eval_sentences.txt", "r", encoding="utf-8") as f:
    sentences = [l.strip() for l in f if l.strip()]

# Создаём папки
os.makedirs("eval_dataset/audio", exist_ok=True)
os.makedirs("eval_dataset/text", exist_ok=True)

references = {}
for i, sentence in enumerate(tqdm(sentences)):
    speaker = random.choice(tts.speakers)
    audio_path = f"eval_dataset/audio/audio_{i:03d}.wav"
    text_path = f"eval_dataset/text/text_{i:03d}.txt"
    
    tts.tts_to_file(
        text=sentence,
        speaker=speaker,
        language="ru",
        file_path=audio_path
    )
    
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(sentence)
    
    references[f"audio_{i:03d}"] = sentence

with open("eval_dataset/references.json", "w", encoding="utf-8") as f:
    json.dump(references, f, ensure_ascii=False, indent=2)

print(f"✅ Готово! {len(references)} пар сгенерировано")

# ====== Ячейка 4: Скачать результат ======
# !zip -r eval_dataset.zip eval_dataset/
# files.download("eval_dataset.zip")
