"""Benchmark Whisper models on the `test_dataset_v1` speech/text dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

OIL_GAS_GLOSSARY = (
    "УПН, ПСН, УПСВ, УКПГ, ГИС, ГРР, КРС, ППД, ЭЦН, АГЗУ, ГРП, ГТМ, АГНКС, АДР, "
    "АСЭЗ, ВВП, ВИЭ, ГКМ, ГМТ, ГПА, ГПЗ, ГПК, ГРС, ГТС, ГЭС, ДМС, ДО, ДТП, ESG, "
    "ЕСУПБ, ЗВ, ИТС, КМН, КПГ, КПО, КПЭ, КриоАЗС, КС, КСГЗ, КСПГ, МГ, МКС, МСП, "
    "МСФО, МТР, НГКМ, НДПИ, НДС, НДТ, НИР, НИОКР, НКО, ОМС, ООС, ООПТ, ОПО, ПГ, "
    "ПНГ, ПХГ, ПЭМ, РО, РСПП, СМИ, СПГ, СТО, СУРиВК, СЦП, СЭМ, ТЭР, ФО, ФОК, ЧС, "
    "ЭТП, НПЗ."
)
GLOSSARY_SET = {word.strip().lower() for word in OIL_GAS_GLOSSARY.replace(".", "").split(",")}
WORD_TO_NUM = {
    "ноль": "0", "один": "1", "одна": "1", "два": "2", "две": "2", "три": "3",
    "четыре": "4", "пять": "5", "шесть": "6", "семь": "7", "восемь": "8",
    "девять": "9", "десять": "10", "двенадцать": "12", "пятнадцать": "15",
    "двадцать": "20", "тридцать шесть": "36", "тридцать": "30", "сорок": "40",
    "пятьдесят": "50", "семьдесят пять": "75", "восемьдесят": "80",
    "сто два": "102", "сто": "100", "двести тридцать семь": "237", "двести": "200",
    "триста": "300", "трёхсот": "300", "тысяча четыреста двадцать": "1420",
    "тысячу": "1000", "две тысячи": "2000", "девяноста пяти": "95",
    "девяноста": "90", "пяти": "5", "восьми": "8", "двенадцати": "12",
    "пятнадцати": "15",
}


def load_dependencies() -> tuple[Any, Any]:
    """Load optional benchmark dependencies.

    Returns:
        tuple[Any, Any]: `WhisperModel` class and `jiwer` module.

    Raises:
        SystemExit: If a required dependency is not installed.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("Install faster-whisper: pip install faster-whisper") from exc
    try:
        import jiwer
    except ImportError as exc:
        raise SystemExit("Install jiwer: pip install jiwer") from exc
    return WhisperModel, jiwer


def get_abbrev_stats(norm_ref: str, norm_hyp: str) -> tuple[int, int]:
    """Count recalled glossary abbreviations in a prediction.

    Args:
        norm_ref (str): Normalized reference text.
        norm_hyp (str): Normalized hypothesis text.

    Returns:
        tuple[int, int]: Found abbreviations and total target abbreviations.

    Raises:
        None.
    """
    reference_words = norm_ref.split()
    hypothesis_words = norm_hyp.split()
    found = 0
    total = 0
    for word in reference_words:
        if word in GLOSSARY_SET:
            total += 1
            if word in hypothesis_words:
                hypothesis_words.remove(word)
                found += 1
    return found, total


def normalize_text(text: str) -> str:
    """Normalize text before metric calculation.

    Args:
        text (str): Raw source text.

    Returns:
        str: Normalized text.

    Raises:
        None.
    """
    normalized = text.lower().replace("ё", "е")
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    normalized = re.sub(r"\\[nrt]", " ", normalized)
    normalized = re.sub(r"[‐‑‒–—−]", "-", normalized)
    normalized = re.sub(r"(?<=\d)[,.](?=\d)", "<decimal>", normalized)
    normalized = re.sub(r"[«»“”„\"'()\[\]{}]", "", normalized)
    normalized = re.sub(r"(?<=[a-zа-я])-(?=[a-zа-я])", " ", normalized)
    normalized = re.sub(r"(?<=[a-zа-я])-(?=[а-яa-z])", " ", normalized)
    normalized = re.sub(r"(?<!\w)-(?!\w)", " ", normalized)
    normalized = re.sub(r"[,\.!?;:]", "", normalized)
    for word, number in sorted(WORD_TO_NUM.items(), key=lambda item: -len(item[0])):
        normalized = re.sub(rf"\b{re.escape(word)}\b", number, normalized)
    normalized = normalized.replace("<decimal>", ".")
    normalized = re.sub(r"\bпроцентов\b|\bпроцента\b|\bпроцент\b", "%", normalized)
    normalized = re.sub(r"(\d)\s*%", r"\1%", normalized)
    normalized = re.sub(r"\bномер\b\s+", "", normalized)
    normalized = re.sub(r"№\s*", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute character error rate without external CER dependencies.

    Args:
        reference (str): Normalized reference text.
        hypothesis (str): Normalized hypothesis text.

    Returns:
        float: Character error rate in the `[0, 1]` range.

    Raises:
        None.
    """
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    if not ref_chars:
        return 0.0
    dp = list(range(len(hyp_chars) + 1))
    for ref_index, ref_char in enumerate(ref_chars, start=1):
        new_dp = [ref_index] + [0] * len(hyp_chars)
        for hyp_index, hyp_char in enumerate(hyp_chars, start=1):
            if ref_char == hyp_char:
                new_dp[hyp_index] = dp[hyp_index - 1]
            else:
                new_dp[hyp_index] = 1 + min(dp[hyp_index], new_dp[hyp_index - 1], dp[hyp_index - 1])
        dp = new_dp
    return dp[-1] / len(ref_chars)


def collect_dataset_items(dataset_dir: str) -> list[tuple[str, Path, str]]:
    """Collect paired audio/text items from `test_dataset_v1`.

    Args:
        dataset_dir (str): Path to the dataset root with `speech` and `text` folders.

    Returns:
        list[tuple[str, Path, str]]: Sample id, audio path, and reference text.

    Raises:
        FileNotFoundError: If required dataset folders are missing.
    """
    root = Path(dataset_dir)
    speech_dir = root / "speech"
    text_dir = root / "text"
    if not speech_dir.is_dir() or not text_dir.is_dir():
        raise FileNotFoundError(f"Expected `speech` and `text` folders in {root}")
    items: list[tuple[str, Path, str]] = []
    for text_path in sorted(text_dir.glob("text_*.txt"), key=lambda path: int(path.stem.split("_")[-1])):
        sample_id = text_path.stem.split("_")[-1]
        audio_path = speech_dir / f"speech_{sample_id}.wav"
        if not audio_path.exists():
            continue
        reference_text = text_path.read_text(encoding="utf-8").strip()
        if reference_text:
            items.append((sample_id, audio_path, reference_text))
    return items


def run_benchmark(dataset_dir: str, model_size: str, use_prompt: bool, results_dir: str) -> None:
    """Run Whisper benchmark on the `test_dataset_v1` dataset.

    Args:
        dataset_dir (str): Dataset root with `speech` and `text` directories.
        model_size (str): Whisper model size.
        use_prompt (bool): Whether to pass the glossary as an initial prompt.
        results_dir (str): Output directory for result JSON files.

    Returns:
        None: This function writes results to disk and prints progress.

    Raises:
        FileNotFoundError: If the dataset structure is invalid.
        SystemExit: If runtime dependencies are missing.
    """
    WhisperModel, jiwer = load_dependencies()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    items = collect_dataset_items(dataset_dir)
    results: list[dict[str, Any]] = []
    all_refs: list[str] = []
    all_hyps: list[str] = []
    total_time = 0.0

    print(f"Running benchmark for {len(items)} files with model={model_size}, prompt={use_prompt}")
    for sample_id, audio_path, reference_text in items:
        start_time = time.time()
        transcribe_kwargs: dict[str, Any] = {"language": "ru"}
        if use_prompt:
            transcribe_kwargs["initial_prompt"] = OIL_GAS_GLOSSARY
        segments, _ = model.transcribe(str(audio_path), **transcribe_kwargs)
        hypothesis_text = " ".join(segment.text for segment in segments).strip()
        inference_time = time.time() - start_time
        total_time += inference_time

        norm_ref = normalize_text(reference_text)
        norm_hyp = normalize_text(hypothesis_text)
        wer_score = jiwer.wer(norm_ref, norm_hyp) if norm_ref else 0.0
        cer_score = compute_cer(norm_ref, norm_hyp)
        abbr_found, abbr_total = get_abbrev_stats(norm_ref, norm_hyp)
        all_refs.append(norm_ref)
        all_hyps.append(norm_hyp)
        results.append({
            "audio_id": sample_id,
            "audio_path": str(audio_path),
            "wer": wer_score,
            "cer": cer_score,
            "abbr_found": abbr_found,
            "abbr_total": abbr_total,
            "inference_time_sec": inference_time,
            "ref_original": reference_text,
            "hyp_original": hypothesis_text,
        })
        print(f"[{sample_id}] time={inference_time:.2f}s wer={wer_score:.2%} cer={cer_score:.2%}")

    macro_wer = sum(item["wer"] for item in results) / len(results) if results else 0.0
    macro_cer = sum(item["cer"] for item in results) / len(results) if results else 0.0
    micro_wer = jiwer.wer(" ".join(all_refs), " ".join(all_hyps)) if all_refs else 0.0
    total_char_errors = sum(compute_cer(ref, hyp) * len(ref.replace(" ", "")) for ref, hyp in zip(all_refs, all_hyps))
    total_ref_chars = sum(len(ref.replace(" ", "")) for ref in all_refs)
    micro_cer = total_char_errors / total_ref_chars if total_ref_chars else 0.0
    total_abbr_found = sum(item["abbr_found"] for item in results)
    total_abbr_target = sum(item["abbr_total"] for item in results)
    summary = {
        "model": model_size,
        "prompt_tuning": use_prompt,
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "wer_macro": macro_wer,
        "wer_micro": micro_wer,
        "cer_macro": macro_cer,
        "cer_micro": micro_cer,
        "abbr_recall": total_abbr_found / total_abbr_target if total_abbr_target else 0.0,
        "abbr_found": total_abbr_found,
        "abbr_total": total_abbr_target,
        "avg_time_sec": total_time / len(results) if results else 0.0,
        "total_files": len(results),
        "details": results,
    }

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    suffix = "_prompted" if use_prompt else "_stock"
    output_path = Path(results_dir) / f"results_{model_size}{suffix}.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path}")


def main() -> None:
    """Parse CLI arguments and run the benchmark.

    Args:
        None.

    Returns:
        None.

    Raises:
        SystemExit: If CLI parsing fails.
    """
    parser = argparse.ArgumentParser(description="Benchmark Whisper on test_dataset_v1")
    parser.add_argument("--dataset_dir", required=True, help="Path to test_dataset_v1 root directory")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3-turbo"],
        help="Whisper model size",
    )
    parser.add_argument("--prompt", action="store_true", help="Enable glossary prompt")
    parser.add_argument("--out_dir", default="whisper_benchmark_results_test_dataset", help="Results directory")
    args = parser.parse_args()
    run_benchmark(args.dataset_dir, args.model, args.prompt, args.out_dir)


if __name__ == "__main__":
    main()
