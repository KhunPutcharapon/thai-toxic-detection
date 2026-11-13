"""เทียบผลสองโมเดลบนชุดทดสอบเดียวกัน ทีละทวีต (เช่น B1 กับ M3)
แต่ละโฟลเดอร์ผลต้องมี metrics.json และ predictions.csv (บันทึกด้วย src.evaluate.save_run)

วิธีรัน:  python compare.py --a results/b1 --b results/m3
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import config
from src.evaluate import METRIC_NAMES, paired_comparison


# อ่าน metrics.json และ predictions.csv ของโมเดลหนึ่งตัว
def load_run(path):
    with open(path / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    preds = pd.read_csv(path / "predictions.csv", dtype={"tweet_id": str})
    return metrics, preds


# จับคู่คำทำนายของสองโมเดลทีละทวีต -> ตารางเมตริก + McNemar + bootstrap -> บันทึก json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, default=config.RESULTS_DIR / "b1")
    ap.add_argument("--b", type=Path, default=config.RESULTS_DIR / "m3")
    args = ap.parse_args()
    name_a, name_b = args.a.name.upper(), args.b.name.upper()

    ma, pa = load_run(args.a)
    mb, pb = load_run(args.b)
    # จับคู่ทีละทวีตตามลำดับของ a — ต้องเป็นชุดทดสอบเดียวกันเป๊ะ
    assert set(pa.tweet_id) == set(pb.tweet_id), "สองโมเดลใช้ชุดทดสอบคนละชุด เทียบกันไม่ได้"
    pb = pb.set_index("tweet_id").loc[pa.tweet_id].reset_index()
    assert (pa.is_toxic.values == pb.is_toxic.values).all(), "ป้ายจริงไม่ตรงกัน"

    print(f"=== {name_b} เทียบ {name_a} บนชุดทดสอบเดียวกัน ({len(pa)} ทวีต) ===")
    print(f"{'':18s}{name_a:>8s}{name_b:>8s}{'ต่าง':>8s}")
    for key, label in METRIC_NAMES:
        print(f"{label:18s}{ma[key]:8.3f}{mb[key]:8.3f}{mb[key] - ma[key]:+8.3f}")

    c = paired_comparison(pa.is_toxic.values, pa.pred.values, pb.pred.values)
    lo, hi = c["acc_diff_ci95"]
    print(f"\nทวีตที่สองโมเดลตัดสินต่างกัน: {c['only_a_right'] + c['only_b_right']} ทวีต "
          f"({name_a} ถูกอย่างเดียว {c['only_a_right']} / {name_b} ถูกอย่างเดียว {c['only_b_right']})")
    print(f"McNemar exact p = {c['mcnemar_exact_p']:.3f}   | accuracy {name_b}-{name_a} = "
          f"{c['acc_diff_b_minus_a']:+.3f}  (bootstrap 95%: {lo:+.3f} ถึง {hi:+.3f})")
    print("ถ้า p >= 0.05 และช่วง 95% คร่อม 0 แปลว่ายังสรุปไม่ได้ว่าสองวิธีต่างกันจริง (ชุดทดสอบ 176 ทวีตเล็ก)")

    out = config.RESULTS_DIR / f"compare_{args.a.name}_vs_{args.b.name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(a=name_a, b=name_b, **c), f, indent=2, ensure_ascii=False)
    print(f"\nบันทึกใน {out}")


if __name__ == "__main__":
    main()
