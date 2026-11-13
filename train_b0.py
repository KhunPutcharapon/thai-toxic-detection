"""B0: ทำซ้ำวิธีของงานเดิม (Sirihattasak et al., 2018) — นับคำ (CountVectorizer) + Logistic Regression
พร้อม keyword baseline ของงานเดิม (ทายว่า toxic ทุกทวีตที่มีคำสำคัญ) และพิมพ์ผลเทียบกับตัวเลขในเปเปอร์

วิธีรัน:  python train_b0.py          -> results/b0/
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"      # ซ่อนคำเตือนของ sklearn (รวมใน worker ของ n_jobs)

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src import config
from src.data import load_raw
from src.evaluate import METRIC_NAMES, compute_metrics, print_metrics, save_run
from src.linear import run_linear_model

OUT_DIR = config.RESULTS_DIR / "b0"
# ตัวเลขที่งานเดิมรายงาน (ใช้เทียบเชิงอ้างอิง — ชุดข้อมูลของเราเหลือ 2,160 จาก 3,300 ทวีต)
PAPER = dict(prec_toxic=0.87, rec_toxic=0.70, f1_toxic=0.78, acc=0.80, kw_baseline_f1_toxic=0.67)


# baseline ของงานเดิม: ทาย toxic ทุกทวีตที่มีคำสำคัญ
def keyword_baseline(test, kws):
    return np.array([int(any(k in t for k in kws)) for t in test["tweet_text"]])


# ฝึก/ทดสอบ B0 + keyword baseline -> บันทึกใน results/b0/ -> พิมพ์ผลเทียบงานเดิม
def main():
    m, test, pred = run_linear_model(CountVectorizer, "B0: Bag-of-Words (CountVectorizer) + Logistic Regression")
    _, kws = load_raw()
    m["kw_baseline_f1_toxic"] = compute_metrics(test["is_toxic"].values, keyword_baseline(test, kws))["f1_toxic"]
    save_run(OUT_DIR, test, pred, m)

    print("\n=== B0 บนชุดทดสอบ 176 ทวีต ===")
    print_metrics(m)
    print(f"\n{'':26s}{'B0':>8s}{'งานเดิม':>9s}")
    for key, name in METRIC_NAMES + [("kw_baseline_f1_toxic", "keyword baseline F1(toxic)")]:
        paper = f"{PAPER[key]:9.2f}" if key in PAPER else f"{'-':>9s}"
        print(f"{name:26s}{m[key]:8.3f}{paper}")
    print(f"\nบันทึกผลใน {OUT_DIR}/")


if __name__ == "__main__":
    main()
