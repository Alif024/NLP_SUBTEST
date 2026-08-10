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

## 2. เทคนิค NLP ที่ต้องใส่ในโค้ด (3 คะแนน)

อ้างอิงตาราง mapping ท้าย [SCHEMA.md](SCHEMA.md) — ทุกเทคนิคต้องผูกกับฟิลด์ผลลัพธ์จริง

- [ ] **Regex & Cleansing** → `clean_text`, `pii_removed`, `price`, `delivery_days`, `variant`, `star_rating`
      (ลบเบอร์โทร / LINE ID / URL / อีเมล / อีโมจิ และ **เก็บของที่ลบไว้แสดง** เพื่อใช้เป็นสัญญาณสแปม)
- [ ] **Tokenization & Normalization** → `tokens`, `tokens_clean`, `normalized_text`
      (`pythainlp.word_tokenize(engine="newmm")`, ลบ stopwords, ยุบคำลากเสียงด้วย `(.)\1{2,}` → `\1`)
- [ ] **Topic Identification** → `aspects` (6 ด้าน), `product_type`, `keywords` (TF-IDF), `topic_id` (LDA)
- [ ] **POS & NER** → `pos_tags`, `brand`, `seller_shop`, `location`, `date_mentioned`
      และใช้ POS ดึงคำคุณศัพท์มาทำ `pros` / `cons`
- [ ] **ตัวคัดกรอง (จุดขายของโปรเจกต์)** → `spam_signals`, `is_ad_spam`, `is_low_quality`,
      `is_duplicate`, `credibility_score`, `review_status`

## 3. โครงสร้างไฟล์ใน Repo (บังคับ)

- [ ] `app.py` — ไฟล์หลักของ Streamlit App
- [ ] `requirements.txt` — runtime เท่านั้น: `streamlit`, `pythainlp`, `spacy`, `pandas`, `scikit-learn`
      (⚠️ **ห้ามใส่ `kagglehub`** — อยู่ใน `requirements-dev.txt` แล้ว)
- [x] `requirements-dev.txt` — `kagglehub`, `pandas` สำหรับเตรียมข้อมูล
- [x] `.gitignore` — `.venv/`, `__pycache__/`, `ref/`
- [x] `scripts/` — `prepare_data.py`, `thai_seed.py`
- [x] `data/` — ไฟล์ข้อมูลทดสอบ 3 ไฟล์
- [ ] `README.md` — ต้องมี 3 ส่วน:
  - [ ] วิธีใช้งาน (รันในเครื่อง + ลิงก์เว็บ)
  - [ ] แนวคิดของ Domain ที่เลือก + ฟิลด์ที่สกัด
  - [ ] **ตัวอย่าง Prompt ที่ใช้สั่ง AI** ตอนเขียนโปรแกรม
- [ ] ไฟล์ข้อมูลทดสอบ (จากข้อ 1)

## 4. หน้าเว็บใช้งานง่าย (1 คะแนน)

**แท็บ 1 — วิเคราะห์รีวิวเดี่ยว**
- [ ] `st.text_area` + ปุ่มประมวลผล + ปุ่ม "ใช้ตัวอย่าง" (ให้อาจารย์กดทดสอบได้ทันที)
- [ ] แสดงป้ายสถานะเด่น ๆ: `review_status` (ผ่าน / น่าสงสัย / สแปม) + `credibility_score`
- [ ] แสดง entity ที่สกัดได้เป็นการ์ด (แบรนด์ / ร้าน / ราคา / วันจัดส่ง / สี-ไซส์)
- [ ] แสดง aspect sentiment เป็นแถว 6 ด้าน พร้อมสีบวก-ลบ + `pros` / `cons`
- [ ] มี expander "ดูขั้นตอนการประมวลผล" โชว์ raw → clean → normalized → tokens → POS
      (สำคัญ: ทำให้เห็นว่าใช้เทคนิคจริง ไม่ใช่เรียก API)

**แท็บ 2 — วิเคราะห์หลายรีวิว**
- [ ] `st.file_uploader` รับ CSV + ปุ่มโหลดไฟล์ตัวอย่างในรีโป
- [ ] ตารางผลลัพธ์ + ตัวกรองตาม `review_status` / aspect
- [ ] กราฟแท่ง aspect sentiment, top pros/cons, ค่าเฉลี่ยวันจัดส่ง
- [ ] ปุ่มดาวน์โหลดผลเป็น CSV

**ทั่วไป**
- [ ] `@st.cache_resource` สำหรับโหลดโมเดล/พจนานุกรม (ไม่งั้นแอปช้ามาก)
- [ ] จัดการ error: ข้อความว่าง, CSV ไม่มีคอลัมน์ที่ต้องการ, ไฟล์ใหญ่เกิน

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
