"""ขั้นที่ 1: สร้างชุดทดสอบ 176 ทวีต และชุดฝึกแบบสมดุล -> data/splits/
รันครั้งเดียว แล้วทุกโมเดล (B1, M3) อ่านไฟล์ชุดเดียวกันนี้

วิธีรัน:  python make_split.py            (ถ้ามีไฟล์อยู่แล้วจะไม่เขียนทับ)
          python make_split.py --force    (สร้างใหม่ทับของเดิม)
"""
import argparse

from src import config
from src.data import load_raw, make_test, make_train, save_splits


# สรุปจำนวนทวีต toxic / non-toxic ของชุดข้อมูล
def describe(name, d):
    return f"{name} {len(d)} ทวีต  toxic={int(d.is_toxic.sum())}  non-toxic={int((1 - d.is_toxic).sum())}"


# สร้างชุดทดสอบและชุดฝึก แล้วบันทึกลง data/splits/ (ไม่เขียนทับถ้าไม่ใส่ --force)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="เขียนทับชุดข้อมูลเดิม")
    args = ap.parse_args()
    if (config.TRAIN_CSV.exists() or config.TEST_CSV.exists()) and not args.force:
        print(f"มีชุดข้อมูลอยู่แล้วใน {config.SPLIT_DIR} — ไม่เขียนทับ (ใช้ --force ถ้าต้องการสร้างใหม่)")
        return

    df, kws = load_raw()
    test, rest = make_test(df, kws)
    print(describe("ชุดทดสอบ", test) + f"  คำสำคัญ={test.keyword.nunique()}")
    print("  (เหลือสำหรับชุดฝึก) " + describe("", rest))

    train, n_dup = make_train(rest, test, kws)
    print(describe("ชุดฝึก", train) + f"  | ไม่มี id ซ้ำกับชุดทดสอบ, ข้อความซ้ำเป๊ะ={n_dup}")

    save_splits(train, test)
    print(f"บันทึกใน {config.SPLIT_DIR}/  (seed={config.SEED}, pool={config.TEST_POOL})")


if __name__ == "__main__":
    main()
