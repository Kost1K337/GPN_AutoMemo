import os
import subprocess
import time

DATASETS = {
    "phrases": "../generation_dataset/eval_dataset_gtts/references.json",
    "meetings": "../generation_dataset/eval_dataset_meetings_gtts/references.json"
}

WHISPER_MODELS = ["small", "medium", "large-v3-turbo"]
OUT_BASE = "whisper_benchmark_results_all"

def run_cmd(cmd):
    print(f"\n" + "="*60)
    print(f"🚀 ЗАПУСК ПОДЗАДАЧИ: {' '.join(cmd)}")
    print("="*60)
    # subprocess.run блокирует выполнение, пока команда не завершится
    subprocess.run(cmd, check=True)

def main():
    start_time = time.time()
    
    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path):
            print(f"⚠️ Датасет {ds_name} не найден по пути: {ds_path}. Шаг пропущен.")
            continue
            
        out_dir = os.path.join(OUT_BASE, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # --- 1. Whisper тестирования ---
        for model in WHISPER_MODELS:
            # Вариант 1: Без словаря (stock)
            run_cmd([
                "python", "benchmark_whisper.py", 
                "--dataset", ds_path, 
                "--model", model, 
                "--out_dir", out_dir
            ])
            # Вариант 2: Со словарем (prompted)
            run_cmd([
                "python", "benchmark_whisper.py", 
                "--dataset", ds_path, 
                "--model", model, 
                "--prompt", 
                "--out_dir", out_dir
            ])

    total_time = time.time() - start_time
    print("\n" + "🏁"*15)
    print(f"✅ ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ!")
    print(f"⏱ ОБЩЕЕ ВРЕМЯ: {total_time/60:.1f} минут")
    print(f"📁 Все JSON-отчеты сохранены в папке: {os.path.abspath(OUT_BASE)}")
    print("🏁"*15 + "\n")

if __name__ == "__main__":
    main()
