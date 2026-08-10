# -*- coding: utf-8 -*-
"""
การวิเคราะห์ระดับชุดข้อมูล: TF-IDF Keywords + LDA Topic Modeling

โมดูลนี้ทำงานได้เฉพาะตอนมีหลายรีวิว เพราะ TF-IDF และ LDA ต้องอาศัยการเทียบ
ระหว่างเอกสาร รีวิวเดี่ยวจึงใช้แค่ POS filtering ในการหา keywords แทน

ความสัมพันธ์กับ aspects.py:
  aspects.py  = Topic Identification แบบกำหนดหัวข้อไว้ล่วงหน้า (6 ด้านที่เราสนใจ)
  corpus.py   = Topic Modeling แบบให้ข้อมูลบอกเองว่ามีกี่กลุ่ม (LDA)
ทั้งสองอย่างเสริมกัน — LDA อาจเผยหัวข้อที่เราไม่ได้ออกแบบไว้
"""

from __future__ import annotations

import numpy as np


def _identity(tokens):
    """ตัวตัดคำหลอก — เราตัดคำมาแล้วจึงส่ง list เข้า sklearn ตรง ๆ"""
    return tokens


def tfidf_keywords(token_lists: list[list[str]], top_n: int = 15) -> list[tuple[str, float]]:
    """
    หาคำสำคัญของทั้งชุดข้อมูลด้วย TF-IDF

    ใช้คะแนนเฉลี่ยข้ามเอกสาร เพื่อหาคำที่ "สำคัญโดยรวม" ไม่ใช่คำที่โดดเฉพาะบางรีวิว
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [t for t in token_lists if t]
    if len(docs) < 2:
        return []

    vec = TfidfVectorizer(
        analyzer=_identity, lowercase=False,
        min_df=2, max_df=0.6,   # ตัดคำที่พบครั้งเดียว และคำที่พบเกิน 60% ของเอกสาร
    )
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:
        return []

    means = np.asarray(matrix.mean(axis=0)).ravel()
    names = vec.get_feature_names_out()
    order = means.argsort()[::-1][:top_n]
    return [(names[i], round(float(means[i]), 4)) for i in order]


def lda_topics(
    token_lists: list[list[str]], n_topics: int = 5, n_words: int = 8, random_state: int = 42
) -> tuple[list[dict], list[int | None]]:
    """
    จัดกลุ่มหัวข้อด้วย LDA (Chapter 4)

    คืนค่า:
      topics      - [{topic_id, words}] คำเด่นของแต่ละกลุ่ม
      assignments - หมายเลขกลุ่มของแต่ละเอกสาร (None ถ้าเอกสารว่าง)
    """
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    index_map = [i for i, t in enumerate(token_lists) if t]
    docs = [token_lists[i] for i in index_map]
    assignments: list[int | None] = [None] * len(token_lists)

    if len(docs) < n_topics * 2:
        return [], assignments

    vec = CountVectorizer(analyzer=_identity, lowercase=False, min_df=2, max_df=0.6)
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:
        return [], assignments
    if matrix.shape[1] < n_topics:
        return [], assignments

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=random_state, learning_method="batch", max_iter=20
    )
    doc_topics = lda.fit_transform(matrix)

    names = vec.get_feature_names_out()
    topics = [
        {"topic_id": t, "words": [names[i] for i in comp.argsort()[::-1][:n_words]]}
        for t, comp in enumerate(lda.components_)
    ]
    for position, doc_index in enumerate(index_map):
        assignments[doc_index] = int(doc_topics[position].argmax())

    return topics, assignments
