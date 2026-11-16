"""เทียบหลายโมเดลพร้อมกันบนชุดทดสอบเดียวกัน (ค่าเริ่มต้น: B0, B1, M3)
ต่อยอดจาก compare.py (ซึ่งเทียบทีละ 2 ตัว) — ใช้ไฟล์ผลชุดเดียวกัน (metrics.json + predictions.csv)

แสดง:
  1) ตารางเมตริกทุกโมเดลเรียงข้างกัน (+ mean ± SD ข้าม seed ถ้ามี เช่น M3)
  2) false positive rate บนทวีต non-toxic (ตอบ H2: คำหยาบแต่ไม่ toxic)
  3) เทียบทีละคู่ทุกคู่: McNemar exact + bootstrap 95% CI
     + ปรับ p ด้วย Holm-Bonferroni เพราะทดสอบหลายคู่พร้อมกัน (กันการเจอ "นัยสำคัญ" โดยบังเอิญ)
  4) ทวีตที่แต่ละโมเดลทายผิด ซ้อนกันแค่ไหน (ผิดทุกตัว / ผิดเฉพาะตัว) — ใช้เลือกตัวอย่างทำ error analysis

วิธีรัน:  python compare_all.py                                  (B0, B1, M3)
          python compare_all.py --runs results/b1 results/m3     (เลือกเอง)
ผลบันทึก: results/compare_all.json            ตัวเลขทั้งหมด (ไว้ให้โค้ดอ่านต่อ)
          results/compare_all_table.csv       ตารางเมตริก โมเดลละคอลัมน์ (ก๊อปลงรายงาน / เปิดใน Excel)
          results/compare_all_pairwise.csv    ตารางเทียบทีละคู่ (McNemar, CI, Holm)
          results/compare_all_errors.csv      ทวีตที่มีอย่างน้อย 1 โมเดลทายผิด (ทำ error analysis)
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.evaluate import METRIC_NAMES, paired_comparison


# อ่าน metrics.json และ predictions.csv ของโมเดลหนึ่งตัว (เหมือน compare.py)
def load_run(path):
    with open(path / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    preds = pd.read_csv(path / "predictions.csv", dtype={"tweet_id": str})
    return metrics, preds


# ปรับ p-value แบบ Holm-Bonferroni: เรียง p จากน้อยไปมาก คูณด้วย (m - ลำดับ) แล้วบังคับให้ไม่ลดลง
def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+",
                    default=[config.RESULTS_DIR / n for n in ("b0", "b1", "m3")])
    args = ap.parse_args()
    names = [p.name.upper() for p in args.runs]

    # --- อ่านผล และจับคู่ทีละทวีตตามลำดับของโมเดลแรก (ต้องเป็นชุดทดสอบเดียวกันเป๊ะ) ---
    runs = [load_run(p) for p in args.runs]
    base = runs[0][1]
    preds = {}
    for name, (_, p) in zip(names, runs):
        assert set(p.tweet_id) == set(base.tweet_id), f"{name} ใช้ชุดทดสอบคนละชุด เทียบไม่ได้"
        p = p.set_index("tweet_id").loc[base.tweet_id].reset_index()
        assert (p.is_toxic.values == base.is_toxic.values).all(), f"ป้ายจริงของ {name} ไม่ตรง"
        preds[name] = p.pred.values
    y = base.is_toxic.values
    n_nontoxic = int((y == 0).sum())
    out = dict(models=names, n_test=len(y))

    # --- 1) ตารางเมตริก ---
    w = 10
    print(f"=== เทียบ {len(names)} โมเดลบนชุดทดสอบเดียวกัน ({len(y)} ทวีต) ===")
    print(f"{'':20s}" + "".join(f"{n:>{w}s}" for n in names))
    for key, label in METRIC_NAMES:
        print(f"{label:20s}" + "".join(f"{m[key]:{w}.3f}" for m, _ in runs))
    print(f"{'FP (จาก ' + str(n_nontoxic) + ')':20s}" + "".join(f"{m['fp']:{w}d}" for m, _ in runs))
    print(f"{'FN':20s}" + "".join(f"{m['fn']:{w}d}" for m, _ in runs))
    out["metrics"] = {n: {k: m[k] for k, _ in METRIC_NAMES + [("fp", ""), ("fn", "")]}
                      for n, (m, _) in zip(names, runs)}

    # mean ± SD ข้าม seed (มีเฉพาะโมเดลที่รันหลาย seed เช่น M3)
    for n, (m, _) in zip(names, runs):
        if "seed_summary" in m:
            s = m["seed_summary"]
            print(f"\n{n} mean ± SD จาก {len(m['seeds'])} seeds: macro-F1 {s['macro_f1']['mean']:.3f} ± "
                  f"{s['macro_f1']['sd']:.3f}, accuracy {s['acc']['mean']:.3f} ± {s['acc']['sd']:.3f}, "
                  f"FP {s['fp']['mean']:.1f} ± {s['fp']['sd']:.1f}")
            out.setdefault("seed_summary", {})[n] = s

    # --- 2) FP rate บนทวีต non-toxic (ทุกทวีตมีคำสำคัญ = "คำหยาบแต่ไม่ toxic") ---
    print(f"\nFP rate บนทวีต non-toxic {n_nontoxic} ทวีต (ยิ่งต่ำยิ่งแยกคำหยาบที่ไม่ toxic ได้ดี):")
    out["fp_rate_nontoxic"] = {}
    for n in names:
        r = float(((preds[n] == 1) & (y == 0)).sum() / n_nontoxic)
        out["fp_rate_nontoxic"][n] = r
        print(f"  {n:6s}{r:6.1%}")

    # --- 3) เทียบทีละคู่ + Holm ---
    pairs = list(combinations(names, 2))
    comps = [paired_comparison(y, preds[a], preds[b]) for a, b in pairs]
    adj = holm([c["mcnemar_exact_p"] for c in comps])
    print(f"\nเทียบทีละคู่ ({len(pairs)} คู่) — p_holm = p ที่ปรับแล้วเพราะทดสอบหลายคู่พร้อมกัน")
    print(f"{'คู่ (b − a)':14s}{'acc ต่าง':>10s}{'95% CI':>20s}{'a ถูก/b ถูก':>12s}{'p':>10s}{'p_holm':>10s}  สรุป")
    out["pairwise"] = []
    for (a, b), c, ph in zip(pairs, comps, adj):
        lo, hi = c["acc_diff_ci95"]
        verdict = "ต่างกันจริง" if ph < 0.05 else "ยังสรุปไม่ได้"
        print(f"{b + ' − ' + a:14s}{c['acc_diff_b_minus_a']:+10.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20s}"
              f"{str(c['only_a_right']) + '/' + str(c['only_b_right']):>12s}"
              f"{c['mcnemar_exact_p']:10.2g}{ph:10.2g}  {verdict}")
        out["pairwise"].append(dict(a=a, b=b, mcnemar_p_holm=ph, **c))

    # --- 4) ทวีตที่ทายผิด ซ้อนกันแค่ไหน ---
    wrong = pd.DataFrame({n: preds[n] != y for n in names})
    print("\nรูปแบบการทายผิด (✗ = ผิด):")
    patterns = wrong.value_counts().sort_index(ascending=False)
    for pat, cnt in patterns.items():
        if not any(pat):
            label = "ถูกทุกโมเดล"
        elif all(pat):
            label = "ผิดทุกโมเดล  <- ยากที่สุด"
        else:
            label = "ผิดเฉพาะ " + ", ".join(n for n, wr in zip(names, pat) if wr)
        print(f"  {'  '.join('✗' if wr else '·' for wr in pat)}   {cnt:4d}   {label}")
    print("  (ลำดับคอลัมน์: " + ", ".join(names) + ")")
    out["error_patterns"] = {"|".join(n for n, wr in zip(names, pat) if wr) or "none": int(cnt)
                             for pat, cnt in patterns.items()}

    # ไฟล์รายทวีต: ไว้เปิดดูตัวอย่างจริงตอนทำ error analysis
    test = pd.read_csv(config.TEST_CSV, dtype={"tweet_id": str}).set_index("tweet_id").loc[base.tweet_id]
    err = test[["tweet_text", "keyword", "is_toxic", "toxic_votes", "nontoxic_votes"]].reset_index()
    for n in names:
        err[f"pred_{n}"] = preds[n]
    err["wrong_by"] = ["|".join(n for n in names if r[n]) for _, r in wrong.iterrows()]
    err = err[err.wrong_by != ""].sort_values(["wrong_by", "is_toxic"])
    err_path = config.RESULTS_DIR / "compare_all_errors.csv"
    err.to_csv(err_path, index=False, encoding="utf-8-sig")

    # --- ตาราง CSV สำหรับรายงาน ---
    # ตารางเมตริก: แถว = เมตริก, คอลัมน์ = โมเดล (+ คอลัมน์ mean ± SD สำหรับโมเดลที่รันหลาย seed)
    rows = [(label, key) for key, label in METRIC_NAMES] + [("FP", "fp"), ("FN", "fn")]
    table = pd.DataFrame({"metric": [r[0] for r in rows]})
    for n, (m, _) in zip(names, runs):
        table[n] = [f"{m[k]:.3f}" if isinstance(m[k], float) else str(m[k]) for _, k in rows]   # ข้อความ: ทศนิยม 3 ตำแหน่ง, FP/FN เป็นจำนวนเต็ม
        if "seed_summary" in m:
            s = m["seed_summary"]
            fmt = lambda k: (f"{s[k]['mean']:.1f} ± {s[k]['sd']:.1f}" if k in ("fp", "fn")
                             else f"{s[k]['mean']:.3f} ± {s[k]['sd']:.3f}")
            table[f"{n} mean ± SD ({len(m['seeds'])} seeds)"] = [fmt(k) for _, k in rows]
    fp_row = {"metric": f"FP rate (non-toxic {n_nontoxic})"}
    fp_row.update({n: f"{out['fp_rate_nontoxic'][n]:.1%}" for n in names})
    table = pd.concat([table, pd.DataFrame([fp_row])], ignore_index=True).fillna("")
    table_path = config.RESULTS_DIR / "compare_all_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")       # utf-8-sig = เปิดใน Excel แล้วภาษาไทยไม่เพี้ยน

    # ตารางเทียบทีละคู่
    pw = pd.DataFrame([dict(
        pair=f"{r['b']} − {r['a']}", acc_diff=round(r["acc_diff_b_minus_a"], 3),
        ci95_low=round(r["acc_diff_ci95"][0], 3), ci95_high=round(r["acc_diff_ci95"][1], 3),
        only_a_right=r["only_a_right"], only_b_right=r["only_b_right"],
        mcnemar_p=float(f"{r['mcnemar_exact_p']:.2g}"), p_holm=float(f"{r['mcnemar_p_holm']:.2g}"),
        significant="yes" if r["mcnemar_p_holm"] < 0.05 else "no") for r in out["pairwise"]])
    pw_path = config.RESULTS_DIR / "compare_all_pairwise.csv"
    pw.to_csv(pw_path, index=False, encoding="utf-8-sig")

    json_path = config.RESULTS_DIR / "compare_all.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nบันทึกใน {json_path}")
    print(f"         {table_path}  (ตารางเมตริก)")
    print(f"         {pw_path}  (เทียบทีละคู่)")
    print(f"         {err_path}  (ทวีตที่มีอย่างน้อย 1 โมเดลทายผิด {len(err)} ทวีต)")


if __name__ == "__main__":
    main()
