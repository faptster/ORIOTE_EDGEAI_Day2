# TGR2026 Edge AI Track 2 · Day 2

ระบบ instance segmentation เมล็ดกาแฟรายเม็ดด้วย YOLO11n-seg (ONNX) บน Raspberry Pi 4 ด้วย CPU ผ่าน ONNX Runtime ไม่ต้องใช้ PyTorch หรือ Ultralytics ตอน inference

## ติดตั้งและทำนาย

แนะนำ Raspberry Pi OS 64 บิต เปิด Terminal ที่โฟลเดอร์ซึ่งมี `infer.py` แล้วติดตั้งด้วย Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir --timeout 120 -r requirements.txt
```

เมื่อวางภาพทดสอบใน `test/` ให้รันคำสั่งมาตรฐาน

```bash
python infer.py --input test/ --weights weights/ --out outputs/
```

- `--input` รับไฟล์ภาพหรือโฟลเดอร์ (อ่านเฉพาะไฟล์ชั้นบนสุดของโฟลเดอร์) รองรับ `.jpg`, `.jpeg`, `.png` โดยไม่สนตัวพิมพ์เล็กใหญ่ เรียงตามชื่อไฟล์ก่อนประมวลผล ชื่อไฟล์และ stem ต้องไม่ซ้ำกัน
- `--weights` รับโฟลเดอร์โมเดล (ดูหัวข้อ weights)
- `--out` โฟลเดอร์ผลลัพธ์ ต้องแยกจาก input และ weights สคริปต์ประมวลผลครบทุกภาพในโฟลเดอร์ชั่วคราวก่อน แล้วจึงแทนที่ผลเดิม ถ้าโฟลเดอร์ out มีไฟล์อื่นที่ไม่ใช่ผลของสคริปต์นี้ จะหยุดโดยไม่ลบอะไร
- `--runtime` (ไม่บังคับ) พิมพ์เวลาประมวลผลต่อภาพ (อ่านภาพ + inference + หลังประมวลผล + เขียนไฟล์) และสรุปค่าเฉลี่ย/สูงสุดท้ายรัน ไม่สร้างไฟล์เพิ่มและไม่เปลี่ยนผลลัพธ์
- สำเร็จจบด้วย exit code 0 ถ้า input/weights ไม่พบ ไม่มีภาพ อ่านภาพไม่ได้ หรือผลผิดรูปแบบ จะแสดง `ERROR: ...` และจบด้วย exit code 1 โดยไม่เขียนผลบางส่วน

ผลลัพธ์

```text
outputs/
├── seg_mask/<stem>.png   # uint16 ช่องเดียว ขนาดเท่าภาพหลังใช้ EXIF orientation, 0 = พื้นหลัง, 1..N = เม็ดที่ 1..N
├── predictions.json      # หนึ่งรายการต่อภาพ: image_name และ objects [{id, class, polygon}]
└── vis/<stem>.jpg        # ภาพ overlay สีตามคลาส เส้นขอบ และป้าย id:class สำหรับดูด้วยตาเท่านั้น
```

## weights

```text
weights/
├── best.onnx     # YOLO11n-seg, input 1×3×320×320, export จาก Ultralytics 8.4.212 (opset 17)
└── config.json   # ตัวเลือกของ infer.py (ไม่บังคับ)
```

`infer.py` เลือกโมเดลตามลำดับ: ชื่อใน `config.json` คีย์ `model` → `best.onnx` → ไฟล์ `.onnx` ไฟล์เดียวในโฟลเดอร์ ชื่อคลาสอ่านจาก metadata ของ ONNX ถ้าชื่อคลาสของโมเดลเป็นหนึ่งใน `black, broken, good, husk, immature, parchment, shell` อยู่แล้วจะใช้ตรง ๆ ไม่เช่นนั้นต้องแปลงผ่าน `class_map` ถ้ามีคลาสใดไม่ได้แปลง สคริปต์จะหยุดพร้อม error

| คีย์ใน config.json | ค่าเริ่มต้น | ความหมาย |
|---|---|---|
| `model` | `best.onnx` | ไฟล์ ONNX ในโฟลเดอร์ weights |
| `class_map` | `{}` | ชื่อคลาสของโมเดล → คลาส Day 2 |
| `conf` | 0.25 | คะแนนคลาสขั้นต่ำ |
| `iou` | 0.7 | IoU threshold ของ NMS |
| `agnostic_nms` | `true` | ทำ NMS ข้ามคลาส ให้หนึ่งเม็ดได้หนึ่ง instance |
| `max_det` | 300 | จำนวน instance สูงสุดต่อภาพ |
| `min_area` | 20 | พื้นที่ instance ขั้นต่ำ (พิกเซลภาพต้นฉบับ) |
| `min_keep` | 0.5 | ทิ้ง instance ที่เหลือพื้นที่ไม่ถึงสัดส่วนนี้หลังหักส่วนที่ทับกับ instance คะแนนสูงกว่า (ถือเป็นตัวซ้ำ) |

`best.onnx` ปัจจุบันมี 12 คลาสภาษาสเปน `config.json` แปลงเป็นคลาส Day 2 ดังนี้

| คลาสโมเดล | คลาส Day 2 |
|---|---|
| negro, agrio | black |
| partido, broca | broken |
| normal, caracolillo, elefante, triangulo | good |
| seca | husk |
| helado | immature |
| concha, oreja | shell |

## การเตรียมภาพและหลังประมวลผล

1. เปิดภาพด้วย Pillow ปรับทิศตาม EXIF แล้วแปลงเป็น RGB
2. Letterbox แบบ Ultralytics: ย่อรักษาสัดส่วนด้วย bilinear ให้ด้านยาวเท่า 320 เติมขอบสีเทา (114, 114, 114) ให้ครบ 320 × 320 หาร 255 เป็น float32 NCHW
3. รัน ONNX Runtime บน CPU ได้ box, คะแนน 12 คลาส และ mask coefficient 32 ค่าต่อ anchor กับ prototype 32 × 80 × 80
4. คัดคะแนน > `conf` แล้ว NMS แบบข้ามคลาส แปลง box กลับพิกัดภาพต้นฉบับ
5. Mask: coefficient × prototype ได้ logit ตัดส่วน padding ออก ขยายด้วย bilinear กลับเป็นความละเอียดภาพต้นฉบับ ตัดนอก box แล้ว threshold ที่ 0 (เหมือน `retina_masks=True` ของ Ultralytics)
6. ให้แต่ละพิกเซลเป็นของ instance เดียว: ไล่จากคะแนนสูงไปต่ำ instance หลังได้เฉพาะพิกเซลที่ยังว่าง
7. ให้ mask ตรงกับ polygon: เก็บชิ้นส่วนต่อเนื่องที่ใหญ่สุดของแต่ละเม็ด เติมรูภายใน แล้วสร้าง polygon ด้วย `cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_NONE)` ค่าใน `seg_mask` ของเม็ดนั้นคือพื้นที่ที่ `cv2.fillPoly` ของ polygon นั้นครอบพอดี ส่วนที่ถูกตัด (ชิ้นแยก ตัวซ้ำ ชิ้นเล็กเกิน) จะรายงานบนหน้าจอของแต่ละภาพ ไม่ทิ้งเงียบ ๆ
8. ก่อนเขียนไฟล์ ตรวจทุกภาพว่า ID เป็น 1..N ตรงกับ PNG คลาสถูกต้อง polygon มีอย่างน้อย 3 จุดอยู่ในภาพ และ polygon ทับ mask พอดีทุกพิกเซล

ไม่มี test-time augmentation ไม่ดาวน์โหลดไฟล์ระหว่าง inference และไม่ปรับโมเดลจากภาพ test

## เทรน

`weights/best.onnx` คือ YOLO11n-seg ที่เทรนด้วย Ultralytics 8.4.212 ที่ imgsz 320 แล้ว export เป็น ONNX (opset 17, batch 1, simplify)

> TODO: อธิบายวิธีรัน `train.py` สำหรับโมเดล segmentation และชุดข้อมูลที่ใช้
