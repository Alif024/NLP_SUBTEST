# ReviewLens TH — ระบบคัดกรองและสกัดข้อมูลรีวิวสินค้าออนไลน์

> Domain ที่เลือก: **รีวิวสินค้า** (แพลตฟอร์ม e-commerce ไทย เช่น Shopee / Lazada / TikTok Shop)
> เอกสารนี้คือการออกแบบ schema ผลลัพธ์ ก่อนลงมือเขียน `app.py`

---

## 1. แนวคิดของระบบ

ปัญหาจริง: รีวิวสินค้าออนไลน์มี **noise** เยอะมาก — รีวิวสแปมแฝงโฆษณา (ทิ้งเบอร์/LINE),
รีวิวปลอมสั้น ๆ ("ดีมากค่ะ" ซ้ำ ๆ), และรีวิวจริงที่ปนข้อมูลส่วนตัวของผู้ซื้อ
ทำให้คนซื้อ "อ่านร้อยรีวิวแล้วยังไม่รู้ว่าสินค้าดีตรงไหน แย่ตรงไหน"

ระบบนี้ทำ 3 อย่างต่อเนื่องกัน:

```
รีวิวดิบ
   │
   ├─ [A] คัดกรอง (Screening)  ── ลบ PII + ตรวจสแปม/รีวิวปลอม → ให้ credibility_score
   │
   ├─ [B] สกัดข้อมูล (Extraction) ── แบรนด์ / ร้าน / ราคา / ระยะเวลาส่ง / รุ่น-สี-ไซส์
   │
   └─ [C] วิเคราะห์ราย Aspect ── แยกเป็น 6 ด้าน + บอกว่าด้านไหนชม ด้านไหนติ
                                   → สรุปเป็น "ข้อดี / ข้อเสีย" ของสินค้า
```

**จุดขาย (creativity):** ไม่ได้บอกแค่ "รีวิวนี้บวก/ลบ" แต่บอกว่า
*ของดีแต่ส่งช้า* หรือ *ส่งไวแต่ของไม่ตรงปก* ซึ่งเป็นสิ่งที่คนซื้อจริงอยากรู้
และคัดรีวิวสแปมทิ้งก่อนสรุป ทำให้ผลสรุปเชื่อถือได้

---

## 2. Schema ผลลัพธ์ (ต่อ 1 รีวิว)

### กลุ่ม A — ข้อมูลตั้งต้น & การทำความสะอาด

| ฟิลด์ | ชนิด | เทคนิคที่ใช้ | คำอธิบาย |
|---|---|---|---|
| `review_id` | str | — | รหัสอ้างอิงรีวิว |
| `raw_text` | str | — | ข้อความดิบก่อนประมวลผล |
| `clean_text` | str | **Regex & Cleansing** | ข้อความหลังลบ URL / เบอร์โทร / LINE ID / อีเมล / อีโมจิ / HTML |
| `pii_removed` | dict | **Regex** | สิ่งที่ถูกลบออก แยกประเภท เช่น `{"phone": ["0812345678"], "line_id": ["@soundshop"]}` |
| `normalized_text` | str | **Normalization** | แก้คำลากเสียง (`ส่งไวมากกกก`→`ส่งไวมาก`), คำแสลง/พิมพ์ผิด (`คับ`→`ครับ`, `มว๊าก`→`มาก`), lowercase อังกฤษ |
| `tokens` | list[str] | **Tokenization** | ตัดคำด้วย `pythainlp.word_tokenize(engine="newmm")` |
| `tokens_clean` | list[str] | **Stopwords** | tokens หลังลบ stopwords ไทย/อังกฤษ และช่องว่าง |
| `pos_tags` | list[tuple] | **POS Tagging** | `[("หูฟัง","NCMN"), ("เสียง","NCMN"), ("ดี","VATT")]` |

### กลุ่ม B — Entity ที่สกัดออกมา

| ฟิลด์ | ชนิด | เทคนิคที่ใช้ | ตัวอย่าง |
|---|---|---|---|
| `brand` | list[str] | **NER (ORGANIZATION)** + gazetteer แบรนด์ | `["Soundcore"]` |
| `product_name` | str \| null | **POS (NCMN/NPRP)** + n-gram | `"หูฟัง Soundcore R50i"` |
| `product_type` | str \| null | **Topic/keyword mapping** | `"หูฟัง"` |
| `seller_shop` | list[str] | **NER (ORGANIZATION)** | `["AnkerThailand"]` |
| `price` | float \| null | **Regex** `(?:฿)?\s*([\d,]+(?:\.\d+)?)\s*(?:บาท\|฿\|THB)` | `890.0` |
| `variant` | dict | **Regex** `(?:สี\|ไซส์\|size\|รุ่น)\s*:?\s*(\S+)` | `{"สี": "ดำ"}` |
| `delivery_days` | int \| null | **Regex** `(\d+)\s*วัน` | `2` |
| `courier` | list[str] | **NER + gazetteer** ขนส่ง | `["Flash"]` |
| `location` | list[str] | **NER (LOCATION)** | `["เชียงใหม่"]` |
| `date_mentioned` | list[str] | **NER (DATE)** / Regex | `["12 ม.ค. 68"]` |
| `star_rating` | float \| null | **Regex** `(\d(?:\.\d)?)\s*(?:ดาว\|/5)` | `4.0` |

### กลุ่ม C — Topic / Aspect & Sentiment

| ฟิลด์ | ชนิด | เทคนิคที่ใช้ | ตัวอย่าง |
|---|---|---|---|
| `aspects` | list[str] | **Topic Identification** | `["shipping", "quality", "packaging", "as_described"]` |
| `aspect_sentiment` | dict | keyword + POS (คำคุณศัพท์) | `{"shipping": "pos", "quality": "pos", "packaging": "neg", "as_described": "neg"}` |
| `pros` | list[str] | **POS (VATT/ADJ)** ใกล้ aspect ที่เป็นบวก | `["ส่งไว", "เสียงดี", "คุ้มราคา"]` |
| `cons` | list[str] | เช่นเดียวกันแต่ฝั่งลบ | `["กล่องบุบ", "ได้สีผิด"]` |
| `overall_sentiment` | str | รวมคะแนนราย aspect | `"mixed"` (`pos` / `neg` / `neutral` / `mixed`) |
| `sentiment_score` | float | −1.0 ถึง 1.0 | `0.15` |
| `keywords` | list[str] | **TF-IDF** + กรองด้วย POS (เอาเฉพาะคำนาม/คุณศัพท์) | `["เสียง", "ส่งไว", "กล่อง"]` |
| `topic_id` | int \| null | **LDA** (Chapter 4) | `2` |

### กลุ่ม D — ผลการคัดกรอง (จุดเด่นของระบบ)

| ฟิลด์ | ชนิด | วิธีคิด | ตัวอย่าง |
|---|---|---|---|
| `spam_signals` | list[str] | รายการสัญญาณที่ตรวจพบ | `["contains_contact", "promo_keyword"]` |
| `is_ad_spam` | bool | มีเบอร์/LINE/URL + คำโปรโมต (`ทักแชท`, `สนใจสั่ง`, `ราคาส่ง`) | `true` |
| `is_low_quality` | bool | tokens_clean < 5 คำ **และ** ไม่ระบุ aspect ใดเลย | `false` |
| `is_duplicate` | bool | ข้อความซ้ำรีวิวอื่น (hash / cosine similarity > 0.9) | `false` |
| `credibility_score` | int | 0–100 หักคะแนนตามสัญญาณข้างบน | `45` |
| `review_status` | str | `"ผ่าน"` / `"น่าสงสัย"` / `"สแปม"` | `"น่าสงสัย"` |

---

## 3. Aspect Taxonomy (6 ด้าน)

| key | ชื่อไทย | คำบ่งชี้ (seed keywords) |
|---|---|---|
| `quality` | คุณภาพสินค้า | ทน, แข็งแรง, พัง, ชำรุด, ของแท้, ของปลอม, งานดี, เนื้อผ้า |
| `as_described` | ความตรงปก | ตรงปก, ไม่ตรงปก, สีเพี้ยน, ผิดไซส์, ได้ไม่ตรง, รูปหลอก |
| `price_value` | ราคา / ความคุ้มค่า | ถูก, แพง, คุ้ม, ไม่คุ้ม, ราคานี้, ลดราคา |
| `shipping` | การจัดส่ง | ส่งไว, ส่งช้า, ขนส่ง, Flash, Kerry, J&T, ตกหล่น, ของหาย |
| `packaging` | บรรจุภัณฑ์ | ห่อมาดี, กล่องบุบ, บับเบิ้ล, แพ็คแน่น, กล่องยับ |
| `seller_service` | บริการร้านค้า | ตอบเร็ว, ไม่ตอบ, แถมของ, บริการดี, ร้านหาย |

> ใช้ seed keywords เป็นตัวตั้งต้น แล้วขยายด้วย TF-IDF จากไฟล์ข้อมูลทดสอบ

---

## 4. ตัวอย่างผลลัพธ์จริง (JSON)

**Input:**
> `ซื้อหูฟัง Soundcore R50i จากร้าน AnkerThailand ราคา 890 บาท ส่งไวมากกกก 2 วันถึงเลย เสียงดีคุ้มราคาา แต่กล่องบุบนิดนึง สั่งสีดำได้สีขาว 🙄 สนใจทักไลน์ @soundshop นะคะ 0812345678`

```json
{
  "review_id": "R001",
  "clean_text": "ซื้อหูฟัง Soundcore R50i จากร้าน AnkerThailand ราคา 890 บาท ส่งไวมากกกก 2 วันถึงเลย เสียงดีคุ้มราคาา แต่กล่องบุบนิดนึง สั่งสีดำได้สีขาว สนใจทักไลน์ นะคะ",
  "pii_removed": { "phone": ["0812345678"], "line_id": ["@soundshop"] },
  "normalized_text": "ซื้อหูฟัง soundcore r50i จากร้าน ankerthailand ราคา 890 บาท ส่งไวมาก 2 วันถึงเลย เสียงดีคุ้มราคา แต่กล่องบุบนิดนึง สั่งสีดำได้สีขาว",

  "brand": ["Soundcore"],
  "product_name": "หูฟัง Soundcore R50i",
  "product_type": "หูฟัง",
  "seller_shop": ["AnkerThailand"],
  "price": 890.0,
  "variant": { "สี_ที่สั่ง": "ดำ", "สี_ที่ได้": "ขาว" },
  "delivery_days": 2,

  "aspects": ["shipping", "quality", "price_value", "packaging", "as_described"],
  "aspect_sentiment": {
    "shipping": "pos",
    "quality": "pos",
    "price_value": "pos",
    "packaging": "neg",
    "as_described": "neg"
  },
  "pros": ["ส่งไว", "เสียงดี", "คุ้มราคา"],
  "cons": ["กล่องบุบ", "ได้สีไม่ตรงที่สั่ง"],
  "overall_sentiment": "mixed",
  "sentiment_score": 0.15,
  "keywords": ["หูฟัง", "เสียง", "ส่งไว", "กล่อง", "สี"],

  "spam_signals": ["contains_contact", "promo_keyword"],
  "is_ad_spam": true,
  "is_low_quality": false,
  "is_duplicate": false,
  "credibility_score": 45,
  "review_status": "น่าสงสัย"
}
```

---

## 5. ผลสรุประดับสินค้า (หน้าที่ 2 ของแอป)

เมื่ออัปโหลดไฟล์รีวิวหลายรายการ ให้สรุปเป็น:

| ฟิลด์ | คำอธิบาย |
|---|---|
| `total_reviews` / `passed` / `flagged` | จำนวนรีวิวทั้งหมด vs. ที่ผ่านการคัดกรอง |
| `aspect_summary` | ตารางราย aspect: `%บวก`, `%ลบ`, จำนวนที่กล่าวถึง |
| `top_pros` / `top_cons` | คำชม/คำติที่พบบ่อยสุด 5 อันดับ |
| `price_range` | ราคาต่ำสุด–สูงสุดที่ผู้รีวิวกล่าวถึง |
| `avg_delivery_days` | ค่าเฉลี่ยระยะเวลาจัดส่ง |
| `top_brands` / `top_shops` | แบรนด์/ร้านที่ถูกกล่าวถึงบ่อย |

พร้อมกราฟแท่ง aspect sentiment + word cloud + ปุ่มดาวน์โหลด CSV

---

## 6. ตารางตรวจสอบว่าใช้เทคนิคครบตามโจทย์

| ข้อกำหนดในโจทย์ | ใช้ตรงไหนใน schema |
|---|---|
| Regex & Cleansing | `clean_text`, `pii_removed`, `price`, `delivery_days`, `variant`, `star_rating` |
| Tokenization & Normalization | `tokens`, `tokens_clean`, `normalized_text` (คำลากเสียง/คำพิมพ์ผิด/stopwords) |
| Topic Identification | `aspects`, `product_type`, `topic_id` (LDA), `keywords` (TF-IDF) |
| POS & NER | `pos_tags`, `brand`, `seller_shop`, `location`, `date_mentioned`, `pros`/`cons` (ดึงคำคุณศัพท์) |
