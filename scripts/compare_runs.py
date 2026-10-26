"""Consolidate ablation runs and print markdown comparison table."""
import csv
import json
from pathlib import Path


def load_log_best(log_path):
    if not Path(log_path).exists():
        return None
    rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
    if not rows:
        return None
    # Find row with lowest val_loss
    best_row = min(rows, key=lambda r: float(r["val_loss"]))
    last_row = rows[-1]
    return {
        "epochs": len(rows),
        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "best_train_loss": float(best_row["train_loss"]),
        "best_act_acc": float(best_row["val_action_acc"]),
        "best_valid": float(best_row["valid_rate"]),
        "best_completion": float(best_row["completion_rate"]),
        "avg_epoch_sec": sum(float(r["epoch_seconds"]) for r in rows) / len(rows),
        "hidden_dim": int(last_row.get("hidden_dim", 256)),
        "lr": float(last_row.get("learning_rate", 0.002)),
        "notes": last_row.get("notes", ""),
    }


def load_eval_report(eval_dir):
    p = Path(eval_dir) / "report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    runs = [
        {
            "name": "Run 1 (Baseline)",
            "log": "logs_qd/train_log.csv",
            "eval_t05": "outputs_qd/eval_t05",
            "eval_t03": "outputs_qd/eval_t03",
        },
        {
            "name": "Run 2 (hidden_dim=384)",
            "log": "logs_qd_h384/train_log.csv",
            "eval_t05": "outputs_qd_h384/eval_t05",
            "eval_t03": "outputs_qd_h384/eval_t03",
        },
        {
            "name": "Run 3 (class_skip=True)",
            "log": "logs_qd_skip/train_log.csv",
            "eval_t05": "outputs_qd_skip/eval_t05",
            "eval_t03": "outputs_qd_skip/eval_t03",
        },
    ]

    print("# Model Improvement Ablation Runs\n")
    print("| Run | Hidden Dim | Params | Epochs | Best Val Loss | Action Acc | Completion (T=0.5) | kNN Recog (T=0.5) | Chair | Bicycle | Bed | Table | Car | House | Time/Epoch |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for r in runs:
        log_info = load_log_best(r["log"])
        if not log_info:
            continue
        eval_t05 = load_eval_report(r["eval_t05"])

        h_dim = log_info["hidden_dim"]
        params = "725k" if h_dim == 384 else "390k"
        best_vl = f"{log_info['best_val_loss']:.4f} (ep {log_info['best_epoch']})"
        acc = f"{log_info['best_act_acc']:.1%}"
        t_sec = f"{log_info['avg_epoch_sec']:.1f}s"

        if eval_t05:
            compl = f"{eval_t05['completion_rate_natural_end']:.1%}"
            knn_all = f"{eval_t05['knn_recognizability_generated']:.1%}"
            pc = eval_t05.get("per_class", {})
            c_chair = f"{pc.get('chair', {}).get('knn_recognized_as_requested', 0):.0%}"
            c_bike = f"{pc.get('bicycle', {}).get('knn_recognized_as_requested', 0):.0%}"
            c_bed = f"{pc.get('bed', {}).get('knn_recognized_as_requested', 0):.0%}"
            c_table = f"{pc.get('table', {}).get('knn_recognized_as_requested', 0):.0%}"
            c_car = f"{pc.get('car', {}).get('knn_recognized_as_requested', 0):.0%}"
            c_house = f"{pc.get('house', {}).get('knn_recognized_as_requested', 0):.0%}"
        else:
            compl = f"{log_info['best_completion']:.1%}"
            knn_all = "Pending eval"
            c_chair = c_bike = c_bed = c_table = c_car = c_house = "—"

        print(f"| {r['name']} | {h_dim} | {params} | {log_info['epochs']} | {best_vl} | {acc} | {compl} | {knn_all} | {c_chair} | {c_bike} | {c_bed} | {c_table} | {c_car} | {c_house} | {t_sec} |")


if __name__ == "__main__":
    main()
