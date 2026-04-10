import argparse
import json
import time
import os
import re
import torch
from pathlib import Path

# Включаем зеркало HuggingFace для загрузки в РФ
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
except ImportError:
    print("Ошибка: не установлены transformers или torch.")
    print("Установите: pip install transformers torchaudio sentencepiece")
    exit(1)

import jiwer

OIL_GAS_GLOSSARY = (
    "УПН, ПСН, УПСВ, УКПГ, ГИС, ГРР, КРС, ППД, ЭЦН, АГЗУ, ГРП, ГТМ, "
    "АГНКС, АДР, АСЭЗ, ВВП, ВИЭ, ГКМ, ГМТ, ГПА, ГПЗ, ГПК, ГРС, ГТС, "
    "ГЭС, ДМС, ДО, ДТП, ЕСГ, ЕСУПБ, ЗВ, ИТС, КМН, КПГ, КПД, КПЭ, "
    "КриоАЗС, КС, КСГЗ, КСПГ, МГ, МКС, МСП, МСФО, МТР, НГКМ, НДПИ, "
    "НДС, НДТ, НИР, НИОКР, НКО, ОМС, ООС, ООПТ, ОПО, ПГ, ПНГ, ПХГ, "
    "ПЭМ, РД, РСПП, СМИ, СПГ, СТО, СУРиВК, СЦП, СЭМ, ТЭР, ФО, ФОК, "
    "ЧС, ЭТП, НПЗ."
)

GLOSSARY_SET = set(w.strip().lower() for w in OIL_GAS_GLOSSARY.replace('.', '').split(','))

def get_abbrev_stats(norm_ref: str, norm_hyp: str) -> tuple[int, int]:
    ref_words = norm_ref.split()
    hyp_words = norm_hyp.split()
    found = 0
    total = 0
    for w in ref_words:
        if w in GLOSSARY_SET:
            total += 1
            if w in hyp_words:
                hyp_words.remove(w)
                found += 1
    return found, total

# Словарь для честной нормализации без шума (используем тот же, что в whisper)
WORD_TO_NUM = {
    'ноль': '0', 'один': '1', 'одна': '1', 'два': '2', 'две': '2',
    'три': '3', 'четыре': '4', 'пять': '5', 'шесть': '6',
    'семь': '7', 'восемь': '8', 'девять': '9', 'десять': '10',
    'двенадцать': '12', 'пятнадцать': '15', 'двадцать': '20',
    'тридцать шесть': '36', 'тридцать': '30', 'сорок': '40',
    'пятьдесят': '50', 'семьдесят пять': '75', 'восемьдесят': '80',
    'сто два': '102', 'сто': '100',
    'двести тридцать семь': '237', 'двести': '200',
    'триста': '300', 'трёхсот': '300',
    'тысяча четыреста двадцать': '1420', 'тысячу': '1000',
    'две тысячи': '2000',
    'девяноста пяти': '95', 'девяноста': '90',
    'пяти': '5', 'восьми': '8', 'двенадцати': '12', 'пятнадцати': '15',
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace('ё', 'е')
    text = re.sub(r'[«»""„\'\(\)\[\]{}]', '', text)
    text = re.sub(r'(?<!\w)-(?!\w)', ' ', text)
    text = re.sub(r'[,\.!?;:]', '', text)
    for word, num in sorted(WORD_TO_NUM.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'\b' + word + r'\b', num, text)
    text = re.sub(r'\bпроцентов\b|\bпроцента\b|\bпроцент\b', '%', text)
    text = re.sub(r'(\d)\s*%', r'\1%', text)
    text = re.sub(r'\bномер\b\s+', '', text)
    text = re.sub(r'№\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_cer(ref: str, hyp: str) -> float:
    ref_chars = list(ref.replace(' ', ''))
    hyp_chars = list(hyp.replace(' ', ''))
    if len(ref_chars) == 0:
        return 0.0
    r, h = len(ref_chars), len(hyp_chars)
    dp = list(range(h + 1))
    for i in range(1, r + 1):
        new_dp = [i] + [0] * h
        for j in range(1, h + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[-1] / len(ref_chars)

def run_benchmark(dataset_json: str, results_dir: str):
    model_id = "ai-sage/GigaAM-v3"
    print(f"Инициализация модели {model_id} на CPU...")
    print("⚠️ Загрузка может занять время. Модель использует custom code (trust_remote_code=True).")
    
    # На CPU используем float32 (float16 / int8 не работают из коробки для transformers без квантования)
    device = "cpu"
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()

    # Pipeline для простоты инференса из аудиофайлов
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=-1 # CPU
    )

    with open(dataset_json, 'r', encoding='utf-8') as f:
        references_data = json.load(f)

    dataset_dir = os.path.dirname(dataset_json)
    
    results = []
    total_time = 0
    all_refs = []
    all_hyps = []

    print(f"\nЗапуск оценки GigaAM (Prompt-Tuning недоступен для архитектуры Conformer)...")

    for audio_id, ref_info in references_data.items():
        audio_file = None
        for ext in ['.mp3', '.wav']:
            potential_path = os.path.join(dataset_dir, "audio", f"{audio_id}{ext}")
            if os.path.exists(potential_path):
                audio_file = potential_path
                break
                
        if not audio_file:
            continue

        ref_text = ref_info.get("text", "")
        if not ref_text:
            continue

        t0 = time.time()
        
        # Inference через pipeline
        with torch.no_grad():
            output = asr_pipe(audio_file)
        
        hyp_text = output["text"].strip()
        
        inf_time = time.time() - t0
        total_time += inf_time

        norm_ref = normalize_text(ref_text)
        norm_hyp = normalize_text(hyp_text)
        
        wer_score = jiwer.wer(norm_ref, norm_hyp) if norm_ref else 0.0
        cer_score = compute_cer(norm_ref, norm_hyp)
        abbr_found, abbr_total = get_abbrev_stats(norm_ref, norm_hyp)

        print(f"\n[{audio_id}] Время: {inf_time:.2f}с | WER: {wer_score:.2%} | CER: {cer_score:.2%} | Аббревиатуры: {abbr_found}/{abbr_total}")
        print(f"  REF: {ref_text}")
        print(f"  HYP: {hyp_text}")

        all_refs.append(norm_ref)
        all_hyps.append(norm_hyp)

        results.append({
            "audio_id": audio_id,
            "wer": wer_score,
            "cer": cer_score,
            "abbr_found": abbr_found,
            "abbr_total": abbr_total,
            "inference_time_sec": inf_time,
            "ref_original": ref_text,
            "hyp_original": hyp_text,
        })

    if all_refs:
        micro_wer = jiwer.wer(" ".join(all_refs), " ".join(all_hyps))
        macro_wer = sum(r["wer"] for r in results) / len(results)
        
        total_char_errors = sum(
            compute_cer(r, h) * len(r.replace(' ', ''))
            for r, h in zip(all_refs, all_hyps)
        )
        total_ref_chars = sum(len(r.replace(' ', '')) for r in all_refs)
        micro_cer = total_char_errors / total_ref_chars if total_ref_chars > 0 else 0.0
        macro_cer = sum(r["cer"] for r in results) / len(results)
    else:
        micro_wer = macro_wer = micro_cer = macro_cer = 0.0

    total_abbr_found = sum(r["abbr_found"] for r in results)
    total_abbr_target = sum(r["abbr_total"] for r in results)
    abbr_recall = total_abbr_found / total_abbr_target if total_abbr_target > 0 else 0.0

    avg_time = total_time / len(results) if results else 0.0

    print("\n" + "=" * 50)
    print("ИТОГИ ТЕСТИРОВАНИЯ GIGAAM-V3")
    print("Аудиофайлов:", len(results))
    print("")
    print(f"WER  macro (среднее по файлам): {macro_wer:.2%}")
    print(f"WER  micro (глобально):         {micro_wer:.2%}")
    print(f"CER  macro (среднее по файлам): {macro_cer:.2%}")
    print(f"CER  micro (глобально):         {micro_cer:.2%}")
    print(f"Аббревиатуры (Recall):          {total_abbr_found}/{total_abbr_target} ({abbr_recall:.2%})")
    print("")
    print(f"Среднее время на файл: {avg_time:.2f} секунд")
    print("=" * 50)

    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"results_gigaam_stock.json")

    summary = {
        "model": "ai-sage/GigaAM-v3",
        "wer_macro": macro_wer,
        "wer_micro": micro_wer,
        "cer_macro": macro_cer,
        "cer_micro": micro_cer,
        "abbr_recall": abbr_recall,
        "abbr_found": total_abbr_found,
        "abbr_total": total_abbr_target,
        "avg_time_sec": avg_time,
        "total_files": len(results),
        "details": results,
    }

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GigaAM Benchmark CPU script")
    parser.add_argument("--dataset", required=True, help="Путь до references.json")
    parser.add_argument("--out_dir", type=str, default="benchmark_results", help="Папка для сохранения результатов")

    args = parser.parse_args()
    run_benchmark(args.dataset, args.out_dir)
