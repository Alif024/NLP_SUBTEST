# สิ่งที่ต้องทำ — แบบทดสอบเก็บคะแนน ครั้งที่ 1 ข้อ 2 (10 คะแนน)

> ที่มา: [ref/Chapter1-5_SubTest1-2.ipynb](ref/Chapter1-5_SubTest1-2.ipynb)
> เป้าหมาย: Web App ประมวลผลข้อความ (ไทย/อังกฤษ) → ขึ้น GitHub → Deploy บน Streamlit Community Cloud
>
> **โปรเจกต์ที่เลือก: ReviewLens TH — ระบบคัดกรองและสกัดข้อมูลรีวิวสินค้าออนไลน์**
> รายละเอียดการออกแบบทั้งหมดอยู่ที่ [SCHEMA.md](SCHEMA.md)

---

## 0. เตรียมการ

- [ ] กรอก **รหัสนักศึกษา** และ **ชื่อ-สกุล** ในไฟล์ notebook
- [x] **เลือกหัวข้อ (Domain)** → **รีวิวสินค้า** (e-commerce ไทย: Shopee / Lazada / TikTok Shop)
- [x] **ออกแบบ schema ผลลัพธ์** → ดู [SCHEMA.md](SCHEMA.md)
  - กลุ่ม A: cleansing (`clean_text`, `pii_removed`, `normalized_text`, `tokens`, `pos_tags`)
  - กลุ่ม B: entity (`brand`, `product_name`, `seller_shop`, `price`, `variant`, `delivery_days`, `courier`, `location`)
  - กลุ่ม C: aspect & sentiment (`aspects`, `aspect_sentiment`, `pros`, `cons`, `keywords`, `topic_id`)
  - กลุ่ม D: การคัดกรอง (`spam_signals`, `is_ad_spam`, `is_duplicate`, `credibility_score`, `review_status`)
- [x] กำหนด **Aspect Taxonomy 6 ด้าน** — quality / as_described / price_value / shipping / packaging / seller_service

## 1. ไฟล์ข้อมูลทดสอบ (1 คะแนน) — ✅ เสร็จแล้ว

**แนวทาง: สองภาษา** — ใช้ข้อมูลจริงภาษาอังกฤษเป็นฐาน + เขียนชุดไทยเองเพื่อเติมเคสที่ข้อมูลจริงไม่มี

- [x] `scripts/prepare_data.py` — ดาวน์โหลด Flipkart dataset (kagglehub) → ทำความสะอาด → สุ่ม stratified
- [x] `scripts/thai_seed.py` — ชุดรีวิวไทย 44 รายการที่เขียนเอง พร้อม `expected_aspects` เป็น ground truth
- [x] `data/reviews_flipkart_en.csv` (500 แถว) / `data/reviews_th.csv` (44) / **`data/reviews_sample.csv` (544 — ไฟล์ที่แอปโหลด)**
- [x] `requirements-dev.txt` แยก `kagglehub` ออกจาก runtime (CSV commit ลงรีโปแล้ว ไม่ต้องโหลดซ้ำตอน deploy)

**คอลัมน์:** `review_id`, `lang`, `source`, `raw_text`, `product_name`, `product_price`, `rate`, `sentiment_label`, `case_tags`, `expected_aspects`

**ผลลัพธ์ที่ได้:** sentiment สมดุล (positive 193 / negative 183 / neutral 158 / mixed 10),
ความยาว median 101 ตัวอักษร (min 2, max 494), ไม่มีค่า null

**ครอบคลุมเคสทดสอบ:** quality 124 · seller_service 124 · delivery_days 119 · packaging 100 ·
as_described 58 · price 56 · elongation 47 · low_quality 67 · ad_spam 4 · pii 9 · duplicate 5 ·
code_switch 4 · typo/slang 6 · html/emoji 3

### สิ่งที่พบจาก dataset (มีผลต่อการเขียนโค้ด)

- ต้นฉบับ 205,052 แถว มี **3 แถวคอลัมน์เลื่อน** (ตรวจจับด้วย `Rate` ที่ไม่ใช่ 1–5) และ **mojibake ใน `product_name`**
- **เครื่องหมายวรรคตอนถูกลบไปหมดแล้ว** → คำติดกัน เช่น `"productfeel cool like acbut trolly should be provided"`
  ต้องมี rule แยกคำติดกันในขั้น cleansing
- **ไม่มี URL เลย (0 แถว) และเบอร์โทรแค่ 2 แถว** → นี่คือเหตุผลที่ต้องมีชุดไทยเขียนเอง ไม่งั้นทดสอบโมดูลคัดกรองไม่ได้
- `Sentiment` label มี noise (บางแถว `rate=4` แต่ label `negative`) → ใช้เป็น ground truth คร่าว ๆ ได้ แต่อย่าอ้างเป็นความจริงสัมบูรณ์
- **ยืนยันแล้วว่า PyThaiNLP ตัด `"มากกกกก"` เป็น `['มาก','กก','กก']`** → ต้อง normalize คำลากเสียง **ก่อน** tokenize เสมอ

### Environment (ทดสอบแล้ว)

- Python 3.14.6 + venv ที่ `.venv/`
- `pythainlp 5.3.5`, `streamlit 1.61`, `spacy 3.8.13`, `scikit-learn 1.9` ติดตั้งได้ทั้งหมด ไม่มีปัญหา wheel

## 2. เทคนิค NLP ที่ต้องใส่ในโค้ด (3 คะแนน) — ✅ เสร็จแล้ว

ตรรกะทั้งหมดอยู่ในแพ็กเกจ `reviewlens/` แยกจาก UI เพื่อให้ทดสอบได้โดยไม่ต้องเปิดเว็บ

- [x] **Regex & Cleansing** → [reviewlens/cleansing.py](reviewlens/cleansing.py)
      ลบ PII/HTML/อีโมจิ + แยกคำอังกฤษที่ติดกันด้วยคลังคำ 1,866 คำ
- [x] **Tokenization & Normalization** → [reviewlens/normalize.py](reviewlens/normalize.py)
      newmm, ยุบคำลากเสียง (ไทย 3+→1, อังกฤษ 3+→2 แล้วเช็คคลังคำ), แก้คำพิมพ์ผิดระดับโทเคน
- [x] **Topic Identification** → [reviewlens/aspects.py](reviewlens/aspects.py) + [corpus.py](reviewlens/corpus.py)
      aspect 6 ด้าน (กำหนดเอง) + TF-IDF + LDA (ให้ข้อมูลบอกเอง)
- [x] **POS & NER** → [reviewlens/extract.py](reviewlens/extract.py)
      pythainlp + spaCy ใช้ป้าย UD ร่วมกัน, gazetteer แบรนด์/ขนส่ง/จังหวัด
- [x] **ตัวคัดกรอง** → [reviewlens/screening.py](reviewlens/screening.py)
      ใช้ PII ที่ลบไปเป็นสัญญาณจับสแปม + ลายนิ้วมือข้อความหารีวิวซ้ำ
- [x] **สคริปต์ประเมินผล** → [scripts/evaluate.py](scripts/evaluate.py)

### ผลการประเมิน (544 รีวิว, 23 ms/รีวิว)

| ตัวชี้วัด | ค่า |
|---|---|
| Aspect precision / recall / F1 | 84.2% / 80.0% / **82.1%** |
| ความถูกต้องของขั้ว | **85.9%** (55/64) |
| จับโฆษณาแฝง / รีวิวซ้ำ / ลบ PII | 4/4 · 3/3 · 2/2 ทุกประเภท |
| ยุบคำลากเสียง / ดึงราคา / ดึงวันจัดส่ง | 47/47 · 37/37 · 55/56 |

### บั๊กที่เจอและแก้ระหว่างทาง (บันทึกไว้อธิบายในรายงานได้)

1. `TYPO_MAP` แทนที่แบบ substring → `"ไม่เป็นไร"` กลายเป็น `"ไม่เป็นอะไร"` — แก้เป็นระดับโทเคน
2. ค้นคำไทยแบบ substring → `"ทน"` แมตช์ใน `"ตัวแทน"`, `"เลย"` เป็นจังหวัด — บังคับให้เริ่มที่ขอบเขตคำ
3. ขอบเขตคำปฏิเสธวัดด้วยระยะตัวอักษร → `"ยังไม่มีปัญหาเลย ทนกว่า"` พลิกขั้ว `"ทน"` ผิด
   — เปลี่ยนเป็นวัดจำนวนคำที่คั่นกลาง (ขั้วถูกขึ้น 82.8% → 85.9%)
4. คำชมกลาง ๆ ถูกผูกกับ aspect คุณภาพ → `"ประทับใจร้านนี้"` นับผิดด้าน — ย้ายไป lexicon กลาง
5. regex ราคา `rs` ไม่มีขอบเขตคำ → `"service centers 2"` กลายเป็นราคา 2 บาท

## 3. โครงสร้างไฟล์ใน Repo (บังคับ) — ✅ เสร็จแล้ว

- [x] `app.py` — Streamlit App 3 แท็บ (รีวิวเดี่ยว / หลายรีวิว / วิธีทำงานของระบบ)
- [x] `requirements.txt` — runtime เท่านั้น + URL โมเดล spaCy (ไม่มี `kagglehub`)
- [x] `.streamlit/config.toml` — ธีมและ `maxUploadSize`
- [x] `reviewlens/` — แพ็กเกจตรรกะ NLP 9 ไฟล์
- [x] `requirements-dev.txt` — `kagglehub`, `pandas` สำหรับเตรียมข้อมูล
- [x] `.gitignore` — `.venv/`, `__pycache__/`, `ref/`
- [x] `scripts/` — `prepare_data.py`, `thai_seed.py`
- [x] `data/` — ไฟล์ข้อมูลทดสอบ 3 ไฟล์
- [x] `README.md` — ครบทั้ง 3 ส่วนที่โจทย์บังคับ:
  - [x] วิธีใช้งาน (ทีละแท็บ + รันในเครื่อง + ขั้นตอน deploy)
  - [x] แนวคิดของ Domain + ตารางฟิลด์ที่สกัดครบทั้ง 4 กลุ่ม
  - [x] **ตัวอย่าง Prompt ที่ใช้สั่ง AI** 5 ตัวอย่าง + บทเรียนที่ได้
- [x] ไฟล์ข้อมูลทดสอบ (จากข้อ 1)
- [ ] ⚠️ **เติม URL เว็บและ GitHub ในหัว README.md หลัง deploy เสร็จ**

## 4. หน้าเว็บใช้งานง่าย (1 คะแนน) — ✅ เสร็จแล้ว

**แท็บ 1 — วิเคราะห์รีวิวเดี่ยว**
- [x] `st.text_area` + ปุ่มวิเคราะห์ + dropdown ตัวอย่าง 6 แบบ (กดทดสอบได้ทันที)
- [x] แถบสถานะ + คะแนนความน่าเชื่อถือ + progress bar
- [x] **เหตุผลที่หักคะแนนทุกข้อ** พร้อมจำนวนคะแนน
- [x] entity เป็นการ์ด (ราคา / วันจัดส่ง / หมวด / ดาว / แบรนด์ / ร้าน / ขนส่ง / สถานที่ / สี-ไซส์)
- [x] aspect 6 ด้านพร้อมสี + **คำที่ใช้ตัดสิน** (ตรวจสอบย้อนกลับได้) + pros/cons
- [x] expander โชว์ raw → clean → normalized → tokens → POS → keywords

**แท็บ 2 — วิเคราะห์หลายรีวิว**
- [x] `st.file_uploader` + ปุ่มใช้ชุดข้อมูลตัวอย่าง + เลือกคอลัมน์/จำนวนได้
- [x] การ์ดสรุป + ตารางราย aspect (ProgressColumn) + กราฟแท่งชม/กลาง/ติ
- [x] คำชม/คำติที่พบบ่อย + TF-IDF + LDA (ปรับจำนวนหัวข้อได้)
- [x] ตารางผลลัพธ์ กรองตามสถานะและ aspect + ปุ่มดาวน์โหลด CSV

**แท็บ 3 — วิธีทำงานของระบบ** (เพิ่มเอง)
- [x] แผนภาพ pipeline + เหตุผลเบื้องหลังการออกแบบ + ตารางคำของ aspect

**ทั่วไป**
- [x] `@st.cache_resource` warm-up โมเดล, `@st.cache_data` แคชผล batch
- [x] จัดการ error: ข้อความว่าง, อ่าน CSV ไม่ได้, จำกัดขนาดอัปโหลด 10 MB
- [x] **ทดสอบด้วย `streamlit.testing.v1.AppTest`** กดปุ่มจริงทุกปุ่ม ไม่มี exception
- [x] เปลี่ยน `use_container_width` → `width="stretch"` (ตัวเก่าถูกกำหนดถอดหลัง 2025-12-31)

## 5. Deploy (2 คะแนน)

- [ ] สร้าง GitHub repository (public หรือให้สิทธิ์เข้าถึงได้)
- [ ] push โค้ดทั้งหมดขึ้น repo
- [ ] Deploy ผ่าน [Streamlit Community Cloud](https://share.streamlit.io/)
- [ ] **ทดสอบเปิด URL จริงจากเครื่องอื่น/มือถือ** ว่าไม่ error
- [ ] ถ้าใช้ spaCy: ต้องโหลดโมเดลให้ได้บน Cloud (ใส่ใน `requirements.txt` เป็น URL ของ wheel หรือใช้ `packages.txt`)

## 6. ส่งงาน

- [ ] เติม **URL หน้าเว็บ Streamlit** ในเซลล์ท้าย notebook
- [ ] เติม **GitHub Repository URL** ในเซลล์ท้าย notebook
- [ ] ส่ง notebook

---

## เกณฑ์คะแนน (รวม 10)

| # | เกณฑ์ | คะแนน |
|---|-------|-------|
| 1 | ไฟล์ข้อมูลที่ใช้ทดสอบ | 1 |
| 2 | ขึ้น GitHub + Streamlit เข้าใช้งานได้จริง | 2 |
| 3 | ความคิดสร้างสรรค์ของหัวข้อ + ออกแบบการสกัดสอดคล้อง | 3 |
| 4 | ใช้เทคนิค NLP ถูกต้องและสอดคล้องหัวข้อ | 3 |
| 5 | หน้าเว็บใช้งานง่าย | 1 |
