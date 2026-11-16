#!/usr/bin/env bash
# รัน B0 + B1 + เทียบผล ในคำสั่งเดียว (ใช้หลังแก้ preprocessing หรือเปลี่ยน config)
# วิธีรัน:  bash run_baselines.sh
set -e                                   # ขั้นไหนพัง หยุดทันที
cd "$(dirname "$0")"
[ -d venv_deepcut ] && source venv_deepcut/bin/activate

python make_split.py                     # มี split อยู่แล้วจะไม่เขียนทับ
python train_b0.py
python train_b1.py
python compare.py --a results/b0 --b results/b1
[ -d results/m3 ] && python compare.py --a results/b1 --b results/m3 || true
echo "เสร็จ — ผลอยู่ใน results/"
