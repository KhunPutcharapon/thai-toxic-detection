"""วัดผล บันทึกผล และเปรียบเทียบสองโมเดลแบบจับคู่ทีละทวีต
ทุกโมเดลบันทึกผลด้วย save_run() ในรูปแบบเดียวกัน => compare.py เทียบโมเดลใดกับโมเดลใดก็ได้"""
import json
import platform

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from . import config

METRIC_NAMES = [("prec_toxic", "precision(toxic)"), ("rec_toxic", "recall(toxic)"),
                ("f1_toxic", "F1(toxic)"), ("acc", "accuracy"), ("macro_f1", "macro-F1")]


# คำนวณ precision/recall/F1 ของคลาส toxic, accuracy, macro-F1 และ confusion matrix
def compute_metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return dict(
        prec_toxic=precision_score(y, p, pos_label=1, zero_division=0),  # ที่ทายว่า toxic ถูกกี่ %
        rec_toxic=recall_score(y, p, pos_label=1),                       # toxic จริงที่จับได้กี่ %
        f1_toxic=f1_score(y, p, pos_label=1),                            # F1 ของคลาส toxic (แบบที่งานเดิมรายงาน)
        acc=accuracy_score(y, p),
        macro_f1=f1_score(y, p, average="macro"),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        predicted_toxic_share=float(np.mean(p)),                         # ชุดทดสอบจริงมี toxic 50%
    )


# ตารางเฉพาะทวีตที่ทายผิด แยก FP (non-toxic ถูกทายเป็น toxic) / FN
def error_table(test, pred):
    err = test.assign(pred=pred)
    err = err[err.pred != err.is_toxic].assign(error_type=lambda d: np.where(d.pred == 1, "FP", "FN"))
    return err[config.SPLIT_COLUMNS + ["pred", "error_type"]]


# เวอร์ชัน Python และไลบรารี ไว้บันทึกเพื่อให้ทำซ้ำได้
def environment_info():
    import sklearn
    return dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__)


# บันทึกผลของโมเดล: metrics.json, predictions.csv (ใช้กับ compare.py), errors.csv
def save_run(out_dir, test, pred, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    test[["tweet_id", "is_toxic", "keyword"]].assign(pred=pred) \
        .to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    error_table(test, pred).to_csv(out_dir / "errors.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


# พิมพ์ confusion matrix และเมตริกหลักบนหน้าจอ
def print_metrics(m):
    print(f"confusion matrix: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}  "
          f"(ทายว่า toxic {m['predicted_toxic_share']:.0%}, ความจริง 50%)")
    for key, name in METRIC_NAMES:
        print(f"  {name:18s}{m[key]:.3f}")


# เทียบสองโมเดลทีละทวีต: McNemar exact test + bootstrap 95% CI ของผลต่าง accuracy
def paired_comparison(y, pred_a, pred_b, seed=config.SEED, n_boot=2000):
    a_wrong, b_wrong = (pred_a != y), (pred_b != y)
    only_a_right = int((~a_wrong & b_wrong).sum())
    only_b_right = int((a_wrong & ~b_wrong).sum())
    disagree = only_a_right + only_b_right
    pval = float(binomtest(min(only_a_right, only_b_right), disagree, 0.5).pvalue) if disagree else 1.0
    rng = np.random.default_rng(seed)
    d_each = (~b_wrong).astype(int) - (~a_wrong).astype(int)             # +1 = b ถูกกว่า, -1 = a ถูกกว่า
    boots = [d_each[rng.integers(0, len(d_each), len(d_each))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(both_right=int((~a_wrong & ~b_wrong).sum()), both_wrong=int((a_wrong & b_wrong).sum()),
                only_a_right=only_a_right, only_b_right=only_b_right, mcnemar_exact_p=pval,
                acc_diff_b_minus_a=float(d_each.mean()), acc_diff_ci95=[float(lo), float(hi)])
