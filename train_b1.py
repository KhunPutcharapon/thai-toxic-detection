"""B1: TF-IDF + Logistic Regression
เปลี่ยนจาก B0 แค่ตัวแปลงข้อความเป็นตัวเลข (CountVectorizer -> TfidfVectorizer) ส่วนอื่นเหมือนกันทุกอย่าง

วิธีรัน:  python train_b1.py          -> results/b1/
"""
import os
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"      # ซ่อนคำเตือนของ sklearn (รวมใน worker ของ n_jobs)

from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.evaluate import print_metrics, save_run
from src.linear import run_linear_model

OUT_DIR = config.RESULTS_DIR / "b1"


# ฝึก/ทดสอบ B1 -> บันทึกใน results/b1/ -> พิมพ์ผล
def main():
    m, test, pred = run_linear_model(TfidfVectorizer, "B1: TF-IDF + Logistic Regression")
    save_run(OUT_DIR, test, pred, m)
    print("\n=== B1 บนชุดทดสอบ 176 ทวีต ===")
    print_metrics(m)
    print(f"\nบันทึกผลใน {OUT_DIR}/")


if __name__ == "__main__":
    main()
