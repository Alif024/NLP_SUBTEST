# -*- coding: utf-8 -*-
"""
ขั้นตอนที่ 2 ของ pipeline: Tokenization & Normalization

ลำดับสำคัญมาก — ต้อง normalize **ก่อน** tokenize เสมอ
เพราะ PyThaiNLP ตัด "ส่งไวมากกกกก" เป็น ['ส่ง','ไว','มาก','กก','กก']
แต่ถ้ายุบคำลากเสียงก่อนจะได้ ['ส่ง','ไว','มาก'] ตามที่ควรเป็น
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import config as cfg
from .cleansing import load_en_vocab

# ---------------------------------------------------------------- โหลดทรัพยากร


@lru_cache(maxsize=1)
def thai_resources():
    """โหลดพจนานุกรมและ stopwords ไทย (แคชไว้ เพราะ thai_words() มี 62,101 คำ)"""
    from pythainlp.corpus import thai_stopwords, thai_words

    return set(thai_words()), set(thai_stopwords()) | cfg.EXTRA_STOPWORDS_TH


@lru_cache(maxsize=1)
def spacy_model():
    """
    โหลดโมเดล spaCy อังกฤษ ถ้าโหลดไม่ได้จะคืน None แล้ว pipeline จะถอยไปใช้ regex + คลังคำแทน
    ทำแบบนี้เพื่อไม่ให้แอปทั้งตัวล่มถ้าโมเดลติดตั้งไม่สำเร็จบน Streamlit Cloud
    """
    try:
        import spacy

        return spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except Exception:
        return None


# ---------------------------------------------------------------- Normalization


def _fix_english_doubles(text: str) -> str:
    """
    หลังยุบตัวซ้ำเหลือ 2 ตัวแล้ว บางคำยังไม่ถูก เช่น "soo" ควรเป็น "so"
    จึงเช็คกับคลังคำอีกชั้น: ถ้าคำยังไม่อยู่ในคลัง ลองยุบตัวซ้ำให้เหลือตัวเดียว
    """
    vocab = load_en_vocab()
    if not vocab:
        return text

    def fix(m: re.Match) -> str:
        word = m.group(0)
        low = word.lower()
        if low in vocab:
            return word
        single = re.sub(r"(.)\1", r"\1", low)
        return single if single in vocab else word

    return re.sub(r"[a-zA-Z]+", fix, text)


def collapse_elongation(text: str) -> str:
    """
    ยุบคำลากเสียง โดยใช้กฎต่างกันตามชนิดอักษร

    ไทย   : ซ้ำ 3+ ตัว -> เหลือ 1  ("มากกกกก" -> "มาก")
            ปลอดภัยเพราะภาษาไทยแทบไม่มีอักษรเดียวกันซ้ำติดกัน 3 ตัว
    อังกฤษ: ซ้ำ 3+ ตัว -> เหลือ 2 แล้วค่อยเช็คคลังคำ
            ("goooood" -> "good" ไม่ใช่ "god", "soooo" -> "so")
    อื่น ๆ : ซ้ำ 3+ ตัว -> เหลือ 1  ("!!!!!" -> "!")
    """
    text = re.sub(r"([ก-๙])\1{2,}", r"\1", text)
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)
    text = re.sub(r"([^ก-๙a-zA-Z\s])\1{2,}", r"\1", text)
    return _fix_english_doubles(text)


def fix_typos(tokens: list[str], lang: str, use_spell_checker: bool = False) -> list[str]:
    """
    แก้คำพิมพ์ผิด/คำแสลงในระดับโทเคน

    ต้องทำหลังตัดคำ ไม่ใช่ก่อน เพราะการ replace ทั้งข้อความจะกินกลางคำ
    เช่นกฎ "ไร" -> "อะไร" จะทำให้ "ไม่เป็นไร" กลายเป็น "ไม่เป็นอะไร"

    use_spell_checker: เปิดใช้ pythainlp.spell เพิ่ม — แม่นขึ้นแต่ช้ามาก
    (~0.1 วิ/คำ) จึงเปิดให้เลือกเฉพาะโหมดวิเคราะห์รีวิวเดี่ยวเท่านั้น
    """
    fixed = [cfg.TYPO_MAP.get(t, t) for t in tokens]

    if use_spell_checker and lang == "th":
        from pythainlp.spell import correct

        words, _ = thai_resources()
        fixed = [
            correct(t) if (len(t) > 2 and t not in words and cfg.THAI_CHAR.search(t)) else t
            for t in fixed
        ]

    return [t for t in (w.strip() for w in fixed) if t]


def normalize(text: str, lang: str) -> str:
    """
    Normalization ระดับตัวอักษร: ยุบคำลากเสียง แล้ว lowercase ส่วนภาษาอังกฤษ

    การแก้คำพิมพ์ผิดไม่ได้ทำที่นี่ เพราะต้องรอให้ตัดคำก่อน (ดู fix_typos)
    """
    text = collapse_elongation(text)
    return re.sub(r"[A-Z]+", lambda m: m.group(0).lower(), text)


# ---------------------------------------------------------------- Tokenization


def tokenize(text: str, lang: str, drop_stopwords: bool = True) -> list[str]:
    """
    ตัดคำตามภาษา

    ไทย   : pythainlp newmm (dictionary-based maximal matching)
    อังกฤษ: spaCy ถ้ามี ไม่งั้นใช้ regex
    """
    if lang == "th":
        from pythainlp import word_tokenize

        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    else:
        nlp = spacy_model()
        if nlp is not None:
            tokens = [t.text for t in nlp(text) if not t.is_space and not t.is_punct]
        else:
            tokens = re.findall(r"[a-zA-Zก-๙]+|\d+", text)

    tokens = [t.strip() for t in tokens if t.strip()]
    return remove_stopwords(tokens, lang) if drop_stopwords else tokens


def remove_stopwords(tokens: list[str], lang: str) -> list[str]:
    """
    ลบ stopwords และคำที่สั้นเกินไปจนไม่มีความหมาย

    ยกเว้นคำใน PROTECTED_WORDS — stopword list ของ PyThaiNLP มีคำอย่าง
    "ส่ง" "ราคา" "ดี" รวมอยู่ด้วย ซึ่งเป็นคำชี้ aspect ที่ระบบนี้ขาดไม่ได้
    """
    _, th_stop = thai_resources()
    out = []
    for t in tokens:
        low = t.lower()
        if t in cfg.PROTECTED_WORDS or low in cfg.PROTECTED_WORDS:
            out.append(t)
            continue
        if low in cfg.STOPWORDS_EN or t in th_stop:
            continue
        if len(t) < 2 and not cfg.THAI_CHAR.search(t):
            continue
        if not re.search(r"[a-zA-Zก-๙\d]", t):
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------- POS Tagging


def pos_tag(tokens: list[str], lang: str) -> list[tuple[str, str]]:
    """
    ติดป้ายชนิดคำ โดยใช้ชุดป้าย Universal Dependencies (NOUN/PROPN/ADJ/VERB/...)
    ทั้งสองภาษา เพื่อให้โค้ดฝั่งที่ใช้ผลลัพธ์ไม่ต้องแยกกรณี

    ไทย   : pythainlp corpus='orchid_ud' ซึ่งแปลง ORCHID -> UD ให้แล้ว
    อังกฤษ: spaCy ใช้ UD อยู่แล้ว
    """
    if not tokens:
        return []

    if lang == "th":
        from pythainlp.tag import pos_tag as th_pos_tag

        try:
            return th_pos_tag(tokens, engine="perceptron", corpus="orchid_ud")
        except Exception:
            return th_pos_tag(tokens)

    nlp = spacy_model()
    if nlp is None:
        return [(t, "X") for t in tokens]

    doc = nlp(" ".join(tokens))
    return [(t.text, t.pos_) for t in doc if not t.is_space]


def content_words(tagged: list[tuple[str, str]], keep=("NOUN", "PROPN", "ADJ", "VERB")) -> list[str]:
    """คัดเฉพาะคำที่มีเนื้อหา ใช้ทำ keywords — ตัดคำเชื่อม คำนำหน้า ออกไป"""
    return [w for w, tag in tagged if tag in keep]
