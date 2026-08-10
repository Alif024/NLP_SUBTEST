# -*- coding: utf-8 -*-
"""
พจนานุกรม แพตเทิร์น และค่าคงที่ทั้งหมดของระบบ

แยกไว้ไฟล์เดียวเพื่อให้ปรับแต่งได้โดยไม่ต้องแตะ logic
ทุก lexicon เป็นสองภาษา (ไทย/อังกฤษ) ในโครงสร้างเดียวกัน
"""

from __future__ import annotations

import re

# ============================================================ Aspect Taxonomy
# แต่ละ aspect มี 3 กลุ่มคำ:
#   cues - คำที่บอกว่ากำลังพูดถึงด้านนี้ (ไม่บอกว่าดีหรือแย่)
#   pos  - คำที่บอกว่าด้านนี้ดี
#   neg  - คำที่บอกว่าด้านนี้แย่
# คำใน pos/neg นับเป็น cue ไปในตัว (พูดถึง "ส่งช้า" ก็คือพูดถึงการจัดส่ง)

ASPECTS: dict[str, dict] = {
    "quality": {
        "label_th": "คุณภาพสินค้า",
        "label_en": "Product Quality",
        "icon": "🔧",
        "cues": [
            "คุณภาพ", "วัสดุ", "งานประกอบ", "เนื้อผ้า", "ตัวสินค้า", "ตัวเครื่อง", "ใช้งาน",
            "quality", "material", "build", "product", "item", "performance", "works",
        ],
        # หมายเหตุ: เก็บเฉพาะคำที่ผูกกับ "คุณภาพสินค้า" จริง ๆ
        # คำชมกลาง ๆ อย่าง "ประทับใจ" "excellent" ย้ายไป GENERAL_POSITIVE แล้ว
        # เพราะรีวิวใช้ชมด้านไหนก็ได้ ("ประทับใจร้านนี้" = บริการ ไม่ใช่คุณภาพ)
        # ปล่อยให้ระยะห่างจาก cue เป็นตัวตัดสินว่าคำชมนั้นพูดถึงด้านใด
        "pos": [
            "ทน", "ทนทาน", "แข็งแรง", "งานดี", "คุณภาพดี", "ใช้ดี", "ใช้ได้ดี", "ใช้ได้",
            "ได้ดี", "ยังดี", "ของแท้", "แน่นหนา", "เนื้อดี", "เสียงดี", "คมชัด", "ทำงานได้ดี",
            "good quality", "great quality", "sturdy", "durable", "solid", "genuine",
            "works well", "working well", "works fine", "nice product", "good product",
        ],
        "neg": [
            "พัง", "ชำรุด", "แตก", "หัก", "ปลอม", "ของปลอม", "ไม่ทน", "หลุด", "เสีย",
            "ขีดข่วน", "มีรอย", "งานหยาบ", "เบี้ยว", "ไม่มีคุณภาพ", "ใช้ไม่ได้", "ใช้งานไม่ได้",
            "poor quality", "bad quality", "low quality", "defective", "defect", "broken",
            "damaged", "stopped working", "not working", "faulty", "cheap quality",
            "duplicate", "fake", "scrap", "malfunction",
        ],
    },
    "as_described": {
        "label_th": "ความตรงปก",
        "label_en": "As Described",
        "icon": "🎯",
        "cues": ["ตรงปก", "ตามรูป", "สเปค", "รายละเอียด", "as described", "as shown", "as advertised", "spec"],
        "pos": [
            "ตรงปก", "ตรงตามรูป", "ตรงตามที่สั่ง", "เหมือนในรูป", "ตรงตามสเปค",
            "as described", "as shown in", "as per description", "same as", "exactly as",
        ],
        "neg": [
            "ไม่ตรงปก", "ไม่ตรงรูป", "สีเพี้ยน", "ผิดไซส์", "ผิดสี", "ผิดรุ่น", "ได้ไม่ตรง",
            "รูปหลอก", "ไม่ตรงตาม", "ไม่เหมือนรูป", "ได้สีผิด", "เล็กกว่าที่ระบุ", "น้อยกว่าที่โฆษณา",
            "not as described", "not as shown", "different from", "wrong item", "wrong size",
            "wrong colour", "wrong color", "wrong product", "misleading", "not same",
            "mismatch", "different product", "not the same",
        ],
    },
    "price_value": {
        "label_th": "ราคา / ความคุ้มค่า",
        "label_en": "Price & Value",
        "icon": "💰",
        "cues": ["ราคา", "บาท", "ตังค์", "เงิน", "price", "cost", "money", "budget", "rs", "rupees"],
        "pos": [
            "คุ้ม", "คุ้มค่า", "คุ้มราคา", "ถูก", "ราคาดี", "ไม่แพง", "ราคาถูก", "ดีเกินราคา", "worth it",
            "value for money", "worth the money", "worth", "cheap", "affordable",
            "reasonable price", "good price", "budget", "best price", "money saver",
        ],
        "neg": [
            "แพง", "ไม่คุ้ม", "แพงไป", "ราคาสูง", "เสียดายตังค์", "เสียเงิน", "ไม่คุ้มราคา",
            "overpriced", "expensive", "costly", "not worth", "waste of money", "too costly",
            "high price", "not worth the money",
        ],
    },
    "shipping": {
        "label_th": "การจัดส่ง",
        "label_en": "Shipping & Delivery",
        "icon": "🚚",
        "cues": [
            "จัดส่ง", "ขนส่ง", "การส่ง", "ส่งของ", "พัสดุ", "ได้ของ", "ของถึง",
            "delivery", "delivered", "shipping", "shipped", "courier", "dispatch", "shipment",
        ],
        "pos": [
            "ส่งไว", "ส่งเร็ว", "ส่งทันใจ", "ได้ของเร็ว", "ถึงไว", "ส่งตรงเวลา", "ส่งฟรี",
            "fast delivery", "quick delivery", "on time", "早", "delivered fast", "prompt delivery",
            "super fast delivery", "timely delivery", "delivered on time", "早い",
        ],
        "neg": [
            "ส่งช้า", "ล่าช้า", "รอนาน", "ของหาย", "ส่งไม่ถึง", "ตกหล่น", "ส่งผิดที่",
            "ส่งไม่สำเร็จ", "ของมาไม่ครบ", "ยังไม่ได้ของ",
            "late delivery", "delayed", "delay", "too late", "slow delivery", "not delivered",
            "never delivered", "lost in transit", "wrong address", "still not received",
        ],
    },
    "packaging": {
        "label_th": "บรรจุภัณฑ์",
        "label_en": "Packaging",
        "icon": "📦",
        "cues": ["แพ็ค", "ห่อ", "กล่อง", "บรรจุภัณฑ์", "บับเบิ้ล", "packaging", "packing", "packed", "box", "carton"],
        "pos": [
            "แพ็คดี", "แพ็คแน่น", "ห่อมาดี", "ห่ออย่างดี", "แน่นหนา", "บับเบิ้ลหนา", "กล่องสวย",
            "ซีลมาดี", "good packaging", "well packed", "nicely packed", "properly packed",
            "sealed", "safe packaging", "excellent packing", "good packing",
        ],
        "neg": [
            "กล่องบุบ", "กล่องยับ", "บุบ", "ยับ", "แพ็คห่วย", "ห่อมาบาง", "แพ็คไม่ดี",
            "กล่องเละ", "กล่องแตก",
            "poor packaging", "bad packing", "damaged box", "box damaged", "torn",
            "not sealed", "open box", "broken box", "worst packing", "poor packing",
        ],
    },
    "seller_service": {
        "label_th": "บริการร้านค้า",
        "label_en": "Seller Service",
        "icon": "🧑‍💼",
        "cues": [
            "ร้าน", "แม่ค้า", "ผู้ขาย", "บริการ", "ติดต่อ", "แชท",
            "seller", "service", "support", "customer care", "response", "vendor",
        ],
        "pos": [
            "ตอบไว", "ตอบเร็ว", "บริการดี", "แถมของ", "ของแถม", "ร้านดี", "ดูแลดี", "ตอบครบ",
            "good service", "responsive", "helpful", "great seller", "good seller",
            "quick response", "excellent service", "thanks to", "thanx", "thank you",
        ],
        "neg": [
            "ไม่ตอบ", "ตอบช้า", "บริการแย่", "ไม่รับผิดชอบ", "ร้านหาย", "เคลมยาก", "ไม่คืนเงิน",
            "poor service", "bad service", "no response", "not responding", "worst service",
            "no support", "not refund", "refuse", "cheated", "scam", "fraud", "complaint",
        ],
    },
}

ASPECT_KEYS = list(ASPECTS)

# ============================================================ คำทั่วไปสำหรับตัดสินขั้ว

# คำแสดงขั้วที่ใช้ชม/ติด้านไหนก็ได้
# ระบบจะโยงคำเหล่านี้เข้ากับ aspect ที่อยู่ใกล้ที่สุดในข้อความ (ดู aspects.py)
# แทนที่จะผูกตายไว้กับ aspect ใด aspect หนึ่ง
GENERAL_POSITIVE = [
    "ดี", "เยี่ยม", "สุดยอด", "ชอบ", "ประทับใจ", "พอใจ", "แนะนำ", "โอเค", "เจ๋ง", "ปัง",
    "สวย", "ดีมาก", "เร็ว", "ไว", "คุ้ม", "เกินคาด", "ถูกใจ", "น่ารัก",
    "good", "great", "nice", "love", "like", "best", "excellent", "happy", "recommend",
    "satisfied", "beautiful", "fine", "ok", "okay", "super", "wonderful", "fantastic",
    "awesome", "amazing", "perfect", "superb", "value", "impressed", "quick", "fast",
]

GENERAL_NEGATIVE = [
    "แย่", "ห่วย", "ผิดหวัง", "เสียใจ", "ไม่ชอบ", "ไม่ดี", "เลวร้าย", "งี่เง่า", "อย่าซื้อ",
    "ไม่ประทับใจ", "ไม่พอใจ", "เสียดาย", "แนะนำว่าอย่า",
    "bad", "poor", "worst", "hate", "terrible", "horrible", "disappointed", "disappointing",
    "awful", "pathetic", "useless", "never buy", "do not buy", "dont buy", "sad", "issue",
    "problem", "waste", "waste of money", "not good", "worse", "regret",
]

# คำปฏิเสธ - ถ้าพบก่อนคำแสดงขั้วในระยะใกล้ จะพลิกขั้ว
NEGATIONS = [
    "ไม่", "ไม่ได้", "ไม่ค่อย", "มิ", "ไร้", "ปราศจาก", "อย่า", "ห้าม",
    "not", "no", "never", "none", "without", "hardly", "cannot", "cant", "dont",
    "doesnt", "didnt", "isnt", "wasnt", "wont", "neither", "nothing",
]
NEGATION_WINDOW = 22  # จำนวนตัวอักษรก่อนหน้าที่จะมองหาคำปฏิเสธ
NEGATION_MAX_GAP = 1  # จำนวนคำที่ยอมให้คั่นระหว่างคำปฏิเสธกับคำเป้าหมาย
#                       ("ไม่ ค่อย ดี" พลิกขั้ว แต่ "ไม่ มีปัญหา เลย ทน" ไม่พลิก)
POLARITY_WINDOW = 55  # รัศมี (ตัวอักษร) ที่นับคำขั้วทั่วไปรอบ ๆ cue ของ aspect

# ============================================================ Regex สำหรับ Cleansing / Extraction

PII_PATTERNS: dict[str, re.Pattern] = {
    # เบอร์โทรไทย: 08x-xxx-xxxx / 0812345678 / 02-xxx-xxxx (คั่นด้วย - เว้นวรรค หรือไม่คั่นเลย)
    "phone": re.compile(r"(?<!\d)0\d[\d\-\s]{7,12}\d(?!\d)"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "url": re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I),
    # LINE ID มี 2 รูปแบบ: ขึ้นต้น @ หรือเขียนนำหน้าว่า line/ไลน์/ไอดี
    "line_id": re.compile(r"(?:(?:line|ไลน์|ไอดีไลน์|ไอดี)\s*(?:id)?\s*[:：]?\s*@?([\w.\-]{3,}))|(?:(?<!\w)@[\w.\-]{3,})", re.I),
    "social": re.compile(r"(?:facebook|fb|ig|instagram|tiktok|whatsapp)\s*[:：]?\s*[\w.\-]{3,}", re.I),
}

HTML_TAG = re.compile(r"<[^>]+>")
HTML_ENTITY = re.compile(r"&[a-z]+;|&#\d+;", re.I)
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF←-⇿⬀-⯿️‍]+"
)

# สกัดค่าตัวเลข - ต้องรันบนข้อความ *ก่อน* ลบ PII เพราะเบอร์โทรอาจถูกลบไปก่อน
PRICE_PATTERN = re.compile(
    r"(?:฿|rs\.?|₹|inr)\s*([\d,]+(?:\.\d+)?)"          # ฿1,290 / Rs. 6195
    r"|([\d,]+(?:\.\d+)?)\s*(?:บาท|฿|thb|rs\.?|₹|rupees?)",  # 890 บาท / 6195 rs
    re.I,
)
DELIVERY_DAYS_PATTERN = re.compile(
    r"(?:ภายใน\s*)?(\d{1,3})\s*(วัน|วันทำการ|days?|day)", re.I
)
STAR_PATTERN = re.compile(r"(\d(?:\.\d)?)\s*(?:ดาว|stars?|/\s*5|เต็ม\s*5)", re.I)
VARIANT_PATTERN = re.compile(
    r"(?:สี|ไซส์|ไซซ์|ขนาด|รุ่น|size|colou?r|model)\s*[:：]?\s*([\wก-๙]{1,15})", re.I
)
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")   # ตัวอักษรเดียวกันซ้ำ 3 ครั้งขึ้นไป
THAI_CHAR = re.compile(r"[ก-๙]")

# ============================================================ Gazetteer (คลังชื่อเฉพาะ)

BRANDS = [
    # แบรนด์ในตลาดไทย
    "Soundcore", "Anker", "JBL", "Sony", "Samsung", "Apple", "Xiaomi", "Redmi", "Realme",
    "Oppo", "Vivo", "OnePlus", "Huawei", "Logitech", "Casio", "Adidas", "Nike", "Lenovo",
    "Asus", "Acer", "HP", "Dell", "Philips", "Panasonic", "Sharp", "Toshiba", "Hitachi",
    "Electrolux", "Delonghi", "Zebra", "Seagate",
    # แบรนด์ที่พบบ่อยในชุดข้อมูล Flipkart
    "Bajaj", "Havells", "Prestige", "Butterfly", "Usha", "Crompton", "Orient", "Voltas",
    "Whirlpool", "Godrej", "Microtek", "Luminous", "Candes", "Maharaja", "Pigeon", "Nova",
    "Boat", "Mi", "Symphony", "Kenstar", "Bluestar", "Lloyd", "Intex", "Amazonbasics",
]

COURIERS = [
    "Flash", "Flash Express", "Kerry", "Kerry Express", "J&T", "JT Express", "DHL", "FedEx",
    "Ninja Van", "Best Express", "SPX", "Shopee Express", "ไปรษณีย์ไทย", "ไปรษณีย์",
    "Ekart", "Bluedart", "Delhivery", "India Post",
]

PLATFORMS = ["Shopee", "Lazada", "TikTok Shop", "Flipkart", "Amazon", "Myntra", "Snapdeal", "JD Central"]

# หมวดสินค้า - ใช้เดา product_type เมื่อไม่มีคอลัมน์ชื่อสินค้า
PRODUCT_TYPES: dict[str, list[str]] = {
    "หูฟัง / ลำโพง": ["หูฟัง", "ลำโพง", "earphone", "headphone", "earbud", "speaker", "airdopes"],
    "มือถือ / แท็บเล็ต": ["มือถือ", "โทรศัพท์", "แท็บเล็ต", "mobile", "phone", "tablet", "smartphone"],
    "คอมพิวเตอร์ / อุปกรณ์": ["โน้ตบุ๊ก", "คีย์บอร์ด", "เมาส์", "laptop", "keyboard", "mouse", "monitor", "printer"],
    "เครื่องใช้ไฟฟ้า": [
        "พัดลม", "หม้อ", "เตา", "ตู้เย็น", "เครื่องซักผ้า", "เครื่องปั่น", "กาต้มน้ำ", "เตารีด",
        "cooler", "fan", "iron", "kettle", "mixer", "grinder", "inverter", "refrigerator",
        "washing machine", "oven", "heater", "vacuum", "purifier",
    ],
    "เสื้อผ้า / แฟชั่น": ["เสื้อ", "กางเกง", "รองเท้า", "กระเป๋า", "นาฬิกา", "shirt", "shoe", "bag", "watch", "jean"],
    "ความงาม / สุขภาพ": ["ครีม", "เซรั่ม", "อาหารเสริม", "วิตามิน", "cream", "serum", "supplement", "lotion"],
    "ของใช้ในบ้าน": ["จาน", "แก้ว", "หมอน", "ผ้าเช็ดตัว", "กล่องเก็บของ", "โคมไฟ", "plate", "pillow", "towel", "lamp"],
}

# ============================================================ สแปม / คำโปรโมต

PROMO_KEYWORDS = [
    "ทักไลน์", "ทักแชท", "ทักมา", "inbox", "สนใจสั่ง", "สนใจทัก", "ราคาส่ง", "รับตัวแทน",
    "ตัวแทนจำหน่าย", "สั่งได้ที่", "ส่งฟรีทั่วประเทศ", "ขายถูกที่สุด", "รับสมัครตัวแทน",
    "ไม่ต้องสต๊อก", "สั่งเลย", "ลด 50", "โปรโมชั่น", "ราคาพิเศษ", "แอดไลน์",
    "contact me", "dm me", "message me", "whatsapp", "click here", "order now",
    "wholesale", "best price guaranteed", "limited offer", "buy now", "visit our",
]

# คำที่พบในรีวิวสั้นทั่วไปที่ไม่ให้ข้อมูลอะไร
GENERIC_PHRASES = [
    "ดี", "ดีค่ะ", "ดีครับ", "ดีมาก", "ดีมากค่ะ", "ดีมากครับ", "โอเค", "ok", "okay",
    "good", "nice", "super", "awesome", "excellent", "best", "very good", "nice product",
    "good product", "fine", "average", "worst", "bad", "ไม่ดี", "แย่",
]

# ============================================================ เกณฑ์การคัดกรอง

SCREENING = {
    "min_tokens": 5,            # น้อยกว่านี้ถือว่าสั้นเกินไป
    "penalty_ad_spam": 70,      # หักคะแนนเมื่อเจอโฆษณา (ต้องพอให้ตกไปถึงระดับ "สแปม")
    "penalty_contact": 15,      # หักเมื่อมีช่องทางติดต่อแต่ไม่เข้าข่ายโฆษณา
    "penalty_low_quality": 35,  # หักเมื่อสั้นและไม่มี aspect (ต้องตกจากระดับ "ผ่าน")
    "penalty_duplicate": 30,    # หักเมื่อข้อความซ้ำกับรีวิวอื่น
    "penalty_generic": 10,      # หักเมื่อเป็นวลีทั่วไปล้วน ๆ
    "penalty_no_aspect": 10,    # หักเมื่อไม่พบ aspect ใดเลย
    "threshold_pass": 70,       # >= ผ่าน
    "threshold_suspect": 40,    # >= น่าสงสัย, ต่ำกว่านี้ = สแปม
}

STATUS_PASS = "ผ่าน"
STATUS_SUSPECT = "น่าสงสัย"
STATUS_SPAM = "สแปม"

# ============================================================ คำพิมพ์ผิด / แสลง

# แก้คำพิมพ์ผิด/แสลงที่พบบ่อยในรีวิวไทย
#
# สำคัญ: ตารางนี้ใช้แทนที่ใน **ระดับโทเคน** เท่านั้น (ดู normalize.fix_typos)
# ห้ามนำไปใช้ str.replace() กับทั้งข้อความ เพราะกฎอย่าง "ไร" -> "อะไร"
# จะไปกินกลางคำ ทำให้ "ไม่เป็นไร" กลายเป็น "ไม่เป็นอะไร"
TYPO_MAP: dict[str, str] = {
    "คับ": "ครับ", "คร้าบ": "ครับ", "ครัช": "ครับ", "คร่ะ": "ค่ะ",
    "มว๊าก": "มาก", "มว้าก": "มาก", "ม๊าก": "มาก", "มั่ก": "มาก",
    "นุง": "นึง", "นิดนุง": "นิดนึง", "เปง": "เป็น",
    "ก้อ": "ก็", "ป่ะ": "ไหม", "ชั้น": "ฉัน", "เค้า": "เขา",
    "ตังค์": "เงิน", "ตัง": "เงิน", "จัย": "ใจ", "งัย": "ไง", "เดี๋ยน": "ดิฉัน",
    "อ่ะ": "", "อะ": "", "ๆ": "",
}

# ============================================================ Stopwords เพิ่มเติม

EXTRA_STOPWORDS_TH = {"ค่ะ", "ครับ", "คะ", "นะ", "จ้า", "จ๊ะ", "ฮะ", "อ่ะ", "เลย", "ๆ", "น่ะ"}

# คำในโดเมนที่ห้ามถูกลบทิ้งตอนลบ stopwords
# จำเป็นเพราะ stopword list ของ PyThaiNLP รวมคำอย่าง "ส่ง" "ราคา" "ดี" ไว้ด้วย
# ซึ่งเป็นคำชี้ aspect ที่ระบบนี้ต้องใช้
PROTECTED_WORDS: set[str] = {
    w
    for aspect in ASPECTS.values()
    for group in ("cues", "pos", "neg")
    for phrase in aspect[group]
    for w in phrase.split()
} | set(GENERAL_POSITIVE) | set(GENERAL_NEGATIVE)
STOPWORDS_EN = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am", "i", "me",
    "my", "we", "our", "you", "your", "he", "she", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "as", "have", "has", "had", "do", "does",
    "did", "will", "would", "can", "could", "should", "there", "here", "what", "which",
    "who", "when", "where", "how", "all", "any", "both", "each", "more", "most", "other",
    "some", "such", "only", "own", "same", "than", "too", "very", "just", "also", "get",
    "got", "one", "two", "after", "before", "again", "s", "t", "m", "re", "ve", "ll", "d",
}
