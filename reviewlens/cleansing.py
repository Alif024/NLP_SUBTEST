# -*- coding: utf-8 -*-
"""
ขั้นตอนที่ 1 ของ pipeline: Regex & Cleansing

สิ่งที่ทำ:
  1. ตรวจภาษาของข้อความ
  2. สกัดและลบข้อมูลติดต่อ (PII) โดย **เก็บสิ่งที่ลบไว้** เพื่อส่งต่อให้โมดูลคัดกรอง
  3. ลบ HTML tag / HTML entity / อีโมจิ
  4. แยกคำภาษาอังกฤษที่ติดกัน (ปัญหาเฉพาะของชุดข้อมูล Flipkart)
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from . import config as cfg

VOCAB_PATH = Path(__file__).resolve().parent.parent / "data" / "en_vocab.txt"


@lru_cache(maxsize=1)
def load_en_vocab() -> dict[str, int]:
    """
    โหลดคลังคำอังกฤษที่สร้างจาก corpus เต็ม (ดู scripts/prepare_data.py)
    คืนค่าเป็น dict คำ -> ลำดับความถี่ (0 = พบบ่อยที่สุด) เพื่อใช้เลือกจุดตัดที่ดีที่สุด
    """
    if not VOCAB_PATH.exists():
        return {}
    words = VOCAB_PATH.read_text(encoding="utf-8").split()
    return {w: i for i, w in enumerate(words)}


def detect_lang(text: str) -> str:
    """
    ตรวจภาษาแบบง่ายด้วยสัดส่วนอักษรไทย
    ใช้เกณฑ์ต่ำ (15%) เพราะรีวิวไทยมักปนอังกฤษเยอะ และ pipeline ไทยรับมือคำอังกฤษได้อยู่แล้ว
    """
    if not text:
        return "en"
    thai = len(cfg.THAI_CHAR.findall(text))
    letters = len(re.findall(r"[ก-๙a-zA-Z]", text))
    if letters == 0:
        return "en"
    return "th" if thai / letters >= 0.15 else "en"


def extract_pii(text: str) -> dict[str, list[str]]:
    """
    สกัดข้อมูลติดต่อออกมาเป็นรายการ แยกตามประเภท

    ลำดับสำคัญ: ต้องหา url และ email ก่อน เพราะทั้งคู่มี @ และจุด
    ซึ่งจะไปชนกับแพตเทิร์นของ LINE ID
    """
    found: dict[str, list[str]] = {}
    remaining = text

    for kind in ("url", "email", "line_id", "social", "phone"):
        hits = []
        for m in cfg.PII_PATTERNS[kind].finditer(remaining):
            hits.append(m.group(0).strip())
        if hits:
            found[kind] = sorted(set(hits))
            # ปิดทับส่วนที่เจอแล้ว เพื่อไม่ให้แพตเทิร์นถัดไปจับซ้ำ
            remaining = cfg.PII_PATTERNS[kind].sub(lambda m: " " * len(m.group(0)), remaining)

    return found


def split_glued_words(text: str, min_len: int = 5) -> str:
    """
    แยกคำอังกฤษที่ติดกัน เช่น "productfeel" -> "product feel"

    ทำไมต้องมี: ชุดข้อมูล Flipkart ถูกลบเครื่องหมายวรรคตอนออกก่อนเผยแพร่
    ทำให้คำท้ายประโยคติดกับคำแรกของประโยคถัดไป
    ("very nice productfeel cool like acbut trolly should be provided")

    วิธีที่ใช้เป็นแบบอนุรักษ์นิยม เพื่อไม่ให้ทำลายคำปกติ:
      - แตะเฉพาะคำที่ *ไม่มี* ในคลังคำ (คำจริงที่พบ >= 50 ครั้งใน 205k แถว)
      - ตัดได้แค่ 2 ท่อน และทั้งสองท่อนต้องอยู่ในคลังคำ
      - ถ้าตัดได้หลายแบบ เลือกแบบที่ทั้งสองคำพบบ่อยที่สุด
    """
    vocab = load_en_vocab()
    if not vocab:
        return text

    def split_token(token: str) -> str:
        low = token.lower()
        if len(low) < min_len or low in vocab or not low.isalpha():
            return token
        best, best_rank = None, None
        for i in range(2, len(low) - 1):
            left, right = low[:i], low[i:]
            if left in vocab and right in vocab:
                rank = vocab[left] + vocab[right]
                if best_rank is None or rank < best_rank:
                    best, best_rank = (left, right), rank
        return f"{best[0]} {best[1]}" if best else token

    return re.sub(r"[a-zA-Z]+", lambda m: split_token(m.group(0)), text)


def clean(text: str, lang: str | None = None) -> dict:
    """
    ทำความสะอาดข้อความ 1 รายการ

    คืนค่า dict ที่มี:
      lang         - ภาษาที่ตรวจได้
      clean_text   - ข้อความหลังลบ PII / HTML / อีโมจิ
      pii_removed  - สิ่งที่ถูกลบ แยกตามประเภท (ส่งต่อให้ screening.py ใช้)
      emoji_count  - จำนวนอีโมจิที่พบ (ใช้ตรวจรีวิวที่มีแต่อีโมจิ)
      had_html     - พบ HTML tag หรือไม่
    """
    text = (text or "").strip()
    lang = lang or detect_lang(text)

    had_html = bool(cfg.HTML_TAG.search(text) or cfg.HTML_ENTITY.search(text))
    stripped = cfg.HTML_TAG.sub(" ", text)
    stripped = cfg.HTML_ENTITY.sub(" ", stripped)

    emoji_count = sum(len(m.group(0)) for m in cfg.EMOJI.finditer(stripped))
    stripped = cfg.EMOJI.sub(" ", stripped)

    pii = extract_pii(stripped)
    for pattern in cfg.PII_PATTERNS.values():
        stripped = pattern.sub(" ", stripped)

    if lang == "en":
        stripped = split_glued_words(stripped)

    stripped = re.sub(r"\s{2,}", " ", stripped).strip()

    return {
        "lang": lang,
        "clean_text": stripped,
        "pii_removed": pii,
        "emoji_count": emoji_count,
        "had_html": had_html,
    }
