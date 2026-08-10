# -*- coding: utf-8 -*-
"""
ReviewLens TH — Streamlit Web Application

หน้าที่ของไฟล์นี้คือ UI เท่านั้น ตรรกะ NLP ทั้งหมดอยู่ในแพ็กเกจ reviewlens/
แยกกันเพื่อให้ทดสอบตรรกะได้โดยไม่ต้องเปิดเว็บ (ดู scripts/evaluate.py)

รันในเครื่อง:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from reviewlens import analyze, analyze_batch, summarize, to_dataframe
from reviewlens.config import ASPECTS, STATUS_PASS, STATUS_SPAM, STATUS_SUSPECT
from reviewlens.corpus import lda_topics, tfidf_keywords

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_CSV = DATA_DIR / "reviews_sample.csv"

st.set_page_config(
    page_title="ReviewLens TH — คัดกรองและสกัดข้อมูลรีวิวสินค้า",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------- สไตล์และตัวช่วยแสดงผล

STATUS_STYLE = {
    STATUS_PASS: ("#0f7b3f", "#e6f4ea", "✅"),
    STATUS_SUSPECT: ("#8a6100", "#fff4e0", "⚠️"),
    STATUS_SPAM: ("#a4262c", "#fdeaea", "🚫"),
}
POLARITY_STYLE = {
    "pos": ("#0f7b3f", "#e6f4ea", "ชม"),
    "neg": ("#a4262c", "#fdeaea", "ติ"),
    "neutral": ("#5a5a5a", "#eeeeee", "กลาง"),
}

st.markdown(
    """
    <style>
      .badge { display:inline-block; padding:2px 10px; border-radius:12px;
               font-size:0.85rem; font-weight:600; margin-right:6px; }
      .chip  { display:inline-block; padding:3px 10px; border-radius:14px;
               background:#f0f2f6; margin:2px 4px 2px 0; font-size:0.85rem; }
      .step  { background:#fafafa; border-left:3px solid #c8c8c8;
               padding:8px 12px; margin-bottom:8px; border-radius:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def badge(text: str, style: tuple[str, str, str]) -> str:
    color, background, _ = style
    return f'<span class="badge" style="color:{color};background:{background}">{text}</span>'


def chips(items) -> str:
    if not items:
        return '<span style="color:#999">—</span>'
    return "".join(f'<span class="chip">{i}</span>' for i in items)


@st.cache_resource(show_spinner="กำลังโหลดโมเดลภาษา (ครั้งแรกใช้เวลาสักครู่)...")
def warm_up() -> str:
    """
    โหลดพจนานุกรมและโมเดลไว้ล่วงหน้า

    จำเป็นเพราะ PyThaiNLP ต้องดาวน์โหลด corpus ครั้งแรก และ spaCy ต้องโหลดโมเดล
    ถ้าไม่ทำตรงนี้ ผู้ใช้จะเจอความหน่วงตอนกดวิเคราะห์ครั้งแรกแทน
    """
    result = analyze("ทดสอบระบบ ส่งไวมาก ของดี test the system")
    return result["review_status"]


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV, keep_default_na=False)


@st.cache_data(show_spinner=False)
def run_batch(texts: tuple[str, ...], ids: tuple[str, ...], names: tuple[str, ...]) -> list[dict]:
    """แคชผลตามเนื้อหาที่ส่งเข้ามา เพื่อไม่ให้ประมวลผลซ้ำเมื่อผู้ใช้เปลี่ยนตัวกรอง"""
    return analyze_batch(list(texts), list(ids), list(names))


# ---------------------------------------------------------------- ส่วนหัว

st.title("🔍 ReviewLens TH")
st.caption(
    "ระบบคัดกรองและสกัดข้อมูลรีวิวสินค้าออนไลน์ (ไทย / อังกฤษ) — "
    "กรองรีวิวสแปม ลบข้อมูลส่วนตัว แล้ววิเคราะห์ว่าสินค้าดีด้านไหน แย่ด้านไหน"
)

with st.sidebar:
    st.header("เกี่ยวกับระบบ")
    st.markdown(
        """
**ปัญหาที่แก้:** รีวิวออนไลน์มีทั้งสแปมโฆษณาแฝง รีวิวปลอมสั้น ๆ
และรีวิวจริงที่ปนเบอร์โทรของผู้ซื้อ ทำให้อ่านร้อยรีวิวแล้วยังไม่รู้ว่า
สินค้าดีตรงไหน แย่ตรงไหน

**ระบบทำ 3 อย่าง**
1. **คัดกรอง** — ลบข้อมูลติดต่อ แล้วใช้สิ่งที่ลบเป็นสัญญาณจับสแปม
2. **สกัดข้อมูล** — แบรนด์ ราคา วันจัดส่ง ขนส่ง สี/ไซส์ สถานที่
3. **วิเคราะห์ราย aspect** — 6 ด้าน บอกว่าด้านไหนชม ด้านไหนติ
        """
    )
    st.divider()
    st.subheader("เทคนิค NLP ที่ใช้")
    st.markdown(
        """
- **Regex & Cleansing** — ลบ PII/HTML/อีโมจิ, แยกคำอังกฤษที่ติดกัน
- **Tokenization & Normalization** — `newmm`, ยุบคำลากเสียง, แก้คำพิมพ์ผิด, ลบ stopwords
- **Topic Identification** — aspect 6 ด้าน + TF-IDF + LDA
- **POS & NER** — `pythainlp` (ไทย), `spaCy` (อังกฤษ), gazetteer
        """
    )
    st.divider()
    use_spell = st.toggle(
        "เปิดตัวตรวจคำสะกดไทย", value=False,
        help="แม่นขึ้นกับคำพิมพ์ผิดที่ไม่อยู่ในตาราง แต่ช้าลงมาก จึงแนะนำเฉพาะรีวิวเดี่ยว",
    )

warm_up()

tab_single, tab_batch, tab_doc = st.tabs(
    ["📝 วิเคราะห์รีวิวเดี่ยว", "📊 วิเคราะห์หลายรีวิว", "📖 วิธีทำงานของระบบ"]
)

# ================================================================ แท็บ 1

SAMPLES = {
    "รีวิวผสม — ของดีแต่ส่งช้า": (
        "ตัวสินค้าใช้ดีนะคะ วัสดุแข็งแรงกว่าที่คิดไว้เยอะ แต่ส่งช้ามาก "
        "รอ 9 วันกว่าจะได้ ขนส่ง Flash ทำของหายรอบนึงด้วย"
    ),
    "สแปมโฆษณาแฝง (มีเบอร์โทร + LINE)": (
        "สนใจสั่งเพิ่มทักไลน์ @deals2024 นะคะ ราคาส่งถูกกว่านี้อีกเยอะ "
        "โทร 081-234-5678 ได้เลยค่ะ รับตัวแทนจำหน่ายด้วย"
    ),
    "รีวิวยาว หลายด้าน": (
        "สั่งจากร้าน SmartHome Official ราคา 2,390 บาท ส่งด้วย J&T ถึงขอนแก่นใน 4 วัน "
        "แพ็คมาดีมากบับเบิ้ลหนา กล่องไม่บุบเลย ตัวเครื่องงานประกอบแน่นหนา "
        "แต่แบตอยู่ได้แค่ 25 นาทีซึ่งน้อยกว่าที่โฆษณาไว้ 40 นาที ร้านตอบแชทช้าไปหน่อย"
    ),
    "คำลากเสียง + คำพิมพ์ผิด": (
        "ของดีคับ ส่งไวคร้าบบบ แต่กล่องบุบนิดนุงคับ ไม่เปงไรร ยังใช้ได้อยู่คับผม"
    ),
    "รีวิวคุณภาพต่ำ": "ดีค่ะ",
    "ภาษาอังกฤษ (Flipkart)": (
        "very best at this price true value for the moneybut delivery was too delayed "
        "took 8 days to deliver and the box was damaged"
    ),
}

with tab_single:
    left, right = st.columns([3, 2])
    with left:
        choice = st.selectbox("เลือกตัวอย่าง หรือพิมพ์ข้อความเอง", ["— พิมพ์เอง —"] + list(SAMPLES))
    with right:
        st.write("")
        st.write("")
        st.caption("ระบบตรวจภาษาให้อัตโนมัติ รองรับไทย อังกฤษ และไทยปนอังกฤษ")

    default_text = SAMPLES.get(choice, "")
    text = st.text_area("ข้อความรีวิว", value=default_text, height=130,
                        placeholder="วางข้อความรีวิวที่ต้องการวิเคราะห์...")

    if st.button("วิเคราะห์", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("กรุณาใส่ข้อความรีวิวก่อน")
        else:
            r = analyze(text, use_spell_checker=use_spell)
            st.session_state["single"] = r

    r = st.session_state.get("single")
    if r:
        st.divider()

        # ---- แถวสถานะการคัดกรอง
        style = STATUS_STYLE[r["review_status"]]
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            st.markdown(
                f"### {style[2]} {badge(r['review_status'], style)}", unsafe_allow_html=True
            )
        c2.metric("ความน่าเชื่อถือ", f"{r['credibility_score']}/100")
        c3.metric("ความรู้สึกรวม", {"positive": "บวก", "negative": "ลบ",
                                     "mixed": "ผสม", "neutral": "กลาง"}[r["overall_sentiment"]])
        c4.metric("ภาษาที่ตรวจได้", "ไทย" if r["lang"] == "th" else "อังกฤษ")
        st.progress(r["credibility_score"] / 100)

        if r["screening_reasons"]:
            with st.expander(f"เหตุผลที่หักคะแนน ({len(r['screening_reasons'])} ข้อ)", expanded=True):
                for reason, penalty in r["screening_reasons"]:
                    st.markdown(f"- {reason}  `−{penalty} คะแนน`")
        if r["pii_removed"]:
            st.info(
                "🔒 ลบข้อมูลติดต่อออกจากข้อความแล้ว: "
                + " · ".join(f"**{k}** {', '.join(v)}" for k, v in r["pii_removed"].items())
            )

        # ---- ข้อมูลที่สกัดได้
        st.subheader("ข้อมูลที่สกัดได้")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("ราคา", f"{r['price']:,.0f}" if r["price"] else "—")
        e2.metric("ระยะเวลาจัดส่ง", f"{r['delivery_days']} วัน" if r["delivery_days"] else "—")
        e3.metric("หมวดสินค้า", r["product_type"] or "—")
        e4.metric("ดาวที่ระบุ", f"{r['star_rating']:g}" if r["star_rating"] else "—")

        f1, f2 = st.columns(2)
        with f1:
            st.markdown("**แบรนด์**  " + chips(r["brand"]), unsafe_allow_html=True)
            st.markdown("**ร้านค้า / แพลตฟอร์ม**  "
                        + chips(r["seller_shop"] + r["platform"]), unsafe_allow_html=True)
            st.markdown("**ขนส่ง**  " + chips(r["courier"]), unsafe_allow_html=True)
        with f2:
            st.markdown("**สถานที่**  " + chips(r["location"]), unsafe_allow_html=True)
            st.markdown("**สี / ไซส์ / รุ่น**  "
                        + chips([f"{k}: {v}" for k, v in r["variant"].items()]),
                        unsafe_allow_html=True)
            st.markdown("**วันที่ที่กล่าวถึง**  " + chips(r["date_mentioned"]), unsafe_allow_html=True)

        # ---- ผลวิเคราะห์ราย aspect
        st.subheader("วิเคราะห์ราย aspect")
        if not r["aspects"]:
            st.warning("ไม่พบการพูดถึงด้านใดของสินค้าเลย — มักเกิดกับรีวิวสั้นหรือรีวิวทั่วไป")
        else:
            for key, spec in ASPECTS.items():
                polarity = r["aspect_sentiment"].get(key)
                if polarity is None:
                    continue
                pstyle = POLARITY_STYLE[polarity]
                col_a, col_b = st.columns([1, 3])
                col_a.markdown(
                    f"{spec['icon']} **{spec['label_th']}** {badge(pstyle[2], pstyle)}",
                    unsafe_allow_html=True,
                )
                col_b.markdown(
                    "ตัดสินจากคำว่า " + chips(r["aspect_evidence"].get(key, [])),
                    unsafe_allow_html=True,
                )

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### 👍 คำชม")
            st.markdown(chips(r["pros"]), unsafe_allow_html=True)
        with g2:
            st.markdown("#### 👎 คำติ")
            st.markdown(chips(r["cons"]), unsafe_allow_html=True)

        # ---- ขั้นตอนการประมวลผล
        with st.expander("🔬 ดูขั้นตอนการประมวลผลทั้งหมด"):
            st.markdown("**1. ข้อความดิบ**")
            st.markdown(f'<div class="step">{r["raw_text"]}</div>', unsafe_allow_html=True)

            st.markdown("**2. หลังทำความสะอาด** — ลบ PII / HTML / อีโมจิ, แยกคำอังกฤษที่ติดกัน")
            st.markdown(f'<div class="step">{r["clean_text"]}</div>', unsafe_allow_html=True)

            st.markdown("**3. หลัง normalize** — ยุบคำลากเสียง, lowercase")
            st.markdown(f'<div class="step">{r["normalized_text"]}</div>', unsafe_allow_html=True)

            st.markdown(f"**4. ตัดคำ** — ได้ {len(r['tokens'])} คำ "
                        f"(เหลือ {len(r['tokens_clean'])} คำหลังลบ stopwords)")
            st.markdown(chips(r["tokens"]), unsafe_allow_html=True)

            st.markdown("**5. ติดป้ายชนิดคำ (POS)**")
            st.dataframe(
                pd.DataFrame(r["pos_tags"], columns=["คำ", "ชนิดคำ"]).T,
                use_container_width=True,
            )

            st.markdown("**6. คำสำคัญที่คัดจาก POS** (เฉพาะคำนาม / คำคุณศัพท์ / คำกริยา)")
            st.markdown(chips(r["keywords"]), unsafe_allow_html=True)

# ================================================================ แท็บ 2

with tab_batch:
    st.markdown(
        "อัปโหลดไฟล์ CSV ที่มีคอลัมน์ข้อความรีวิว หรือกดใช้ชุดข้อมูลตัวอย่างที่มากับระบบ "
        "(544 รีวิว: Flipkart 500 + ไทยที่เขียนเอง 44)"
    )

    up1, up2 = st.columns([2, 1])
    with up1:
        uploaded = st.file_uploader("ไฟล์ CSV", type=["csv"])
    with up2:
        st.write("")
        use_sample = st.button("📂 ใช้ชุดข้อมูลตัวอย่าง", use_container_width=True)

    source: pd.DataFrame | None = None
    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded, keep_default_na=False)
        except Exception as exc:
            st.error(f"อ่านไฟล์ไม่สำเร็จ: {exc}")
    elif use_sample or st.session_state.get("used_sample"):
        st.session_state["used_sample"] = True
        source = load_sample()

    if source is not None and not source.empty:
        # เดาคอลัมน์ข้อความให้อัตโนมัติ แล้วให้ผู้ใช้แก้ได้
        candidates = [c for c in source.columns if source[c].dtype == object]
        default_col = next(
            (c for c in ("raw_text", "review", "text", "Summary", "comment") if c in source.columns),
            candidates[0] if candidates else source.columns[0],
        )
        col1, col2 = st.columns([2, 1])
        text_col = col1.selectbox(
            "คอลัมน์ที่เก็บข้อความรีวิว", source.columns,
            index=list(source.columns).index(default_col),
        )
        limit = col2.slider("จำนวนรีวิวที่วิเคราะห์", 20, min(len(source), 800),
                            min(len(source), 300), step=20)

        if st.button("เริ่มวิเคราะห์", type="primary", use_container_width=True):
            subset = source.head(limit)
            ids = (subset["review_id"].astype(str).tolist()
                   if "review_id" in subset.columns
                   else [f"R{i:04d}" for i in range(1, len(subset) + 1)])
            names = (subset["product_name"].astype(str).tolist()
                     if "product_name" in subset.columns else [""] * len(subset))
            with st.spinner(f"กำลังวิเคราะห์ {len(subset)} รีวิว..."):
                st.session_state["batch"] = run_batch(
                    tuple(subset[text_col].astype(str)), tuple(ids), tuple(names)
                )

    results = st.session_state.get("batch")
    if results:
        s = summarize(results)
        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("รีวิวทั้งหมด", s["total"])
        m2.metric("✅ ผ่าน", s["passed"], f"{s['passed'] / s['total']:.0%}")
        m3.metric("⚠️ น่าสงสัย", s["suspect"], f"{s['suspect'] / s['total']:.0%}")
        m4.metric("🚫 สแปม", s["spam"], f"-{s['spam'] / s['total']:.0%}", delta_color="inverse")

        n1, n2, n3 = st.columns(3)
        n1.metric("ราคาที่กล่าวถึง",
                  f"{s['price_range'][0]:,.0f} – {s['price_range'][1]:,.0f}"
                  if s["price_range"] else "—")
        n2.metric("ระยะเวลาจัดส่งเฉลี่ย",
                  f"{s['avg_delivery_days']} วัน" if s["avg_delivery_days"] else "—")
        n3.metric("ความรู้สึกผสม (mixed)", s["sentiment_counts"]["mixed"])

        # ---- สรุปราย aspect
        st.subheader("สรุปราย aspect")
        st.caption("นับเฉพาะรีวิวที่ผ่านการคัดกรอง (ไม่รวมที่ถูกจัดเป็นสแปม)")
        if s["aspect_summary"]:
            aspect_df = pd.DataFrame(s["aspect_summary"])
            st.dataframe(
                aspect_df[["aspect", "mentions", "pos", "neg", "neutral", "pos_pct"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "aspect": "ด้าน",
                    "mentions": st.column_config.NumberColumn("ถูกพูดถึง"),
                    "pos": st.column_config.NumberColumn("ชม"),
                    "neg": st.column_config.NumberColumn("ติ"),
                    "neutral": st.column_config.NumberColumn("กลาง"),
                    "pos_pct": st.column_config.ProgressColumn(
                        "สัดส่วนคำชม", format="%.0f%%", min_value=0, max_value=100
                    ),
                },
            )
            st.bar_chart(aspect_df.set_index("aspect")[["pos", "neutral", "neg"]],
                         color=["#2e7d32", "#9e9e9e", "#c62828"], height=280)
        else:
            st.info("ไม่พบ aspect ในชุดข้อมูลนี้")

        # ---- คำชม / คำติ ที่พบบ่อย
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("#### 👍 คำชมที่พบบ่อย")
            if s["top_pros"]:
                st.dataframe(pd.DataFrame(s["top_pros"], columns=["คำ", "จำนวน"]),
                             hide_index=True, use_container_width=True)
            else:
                st.caption("—")
        with p2:
            st.markdown("#### 👎 คำติที่พบบ่อย")
            if s["top_cons"]:
                st.dataframe(pd.DataFrame(s["top_cons"], columns=["คำ", "จำนวน"]),
                             hide_index=True, use_container_width=True)
            else:
                st.caption("—")

        # ---- TF-IDF และ LDA
        st.subheader("คำสำคัญและหัวข้อที่ค้นพบเอง")
        st.caption(
            "TF-IDF หาคำสำคัญของทั้งชุดข้อมูล ส่วน LDA จัดกลุ่มหัวข้อโดยไม่ต้องกำหนดล่วงหน้า "
            "— ต่างจาก aspect 6 ด้านที่เรากำหนดไว้เอง จึงอาจเผยหัวข้อที่ออกแบบไม่ครอบคลุม"
        )
        token_lists = [r["tokens_clean"] for r in results]
        k1, k2 = st.columns([1, 2])
        with k1:
            st.markdown("**TF-IDF Keywords**")
            keywords = tfidf_keywords(token_lists, top_n=15)
            if keywords:
                st.dataframe(pd.DataFrame(keywords, columns=["คำ", "คะแนน"]),
                             hide_index=True, use_container_width=True, height=380)
            else:
                st.caption("ข้อมูลน้อยเกินไป")
        with k2:
            st.markdown("**LDA Topic Modeling**")
            n_topics = st.slider("จำนวนหัวข้อ", 3, 8, 5, key="lda_k")
            topics, _ = lda_topics(token_lists, n_topics=n_topics)
            if topics:
                for t in topics:
                    st.markdown(f"**หัวข้อ {t['topic_id'] + 1}** " + chips(t["words"]),
                                unsafe_allow_html=True)
            else:
                st.caption("ข้อมูลน้อยเกินไปสำหรับ LDA")

        # ---- ตารางผลลัพธ์
        st.subheader("ผลลัพธ์รายรีวิว")
        table = to_dataframe(results)
        t1, t2 = st.columns(2)
        status_filter = t1.multiselect("กรองตามสถานะ", table["review_status"].unique(),
                                       default=list(table["review_status"].unique()))
        aspect_filter = t2.selectbox("กรองตาม aspect",
                                     ["ทั้งหมด"] + [f"{v['icon']} {v['label_th']}"
                                                     for v in ASPECTS.values()])
        view = table[table["review_status"].isin(status_filter)]
        if aspect_filter != "ทั้งหมด":
            key = [k for k, v in ASPECTS.items()
                   if f"{v['icon']} {v['label_th']}" == aspect_filter][0]
            view = view[view[f"aspect_{key}"] != ""]

        st.dataframe(view, use_container_width=True, hide_index=True, height=420)
        st.caption(f"แสดง {len(view)} จาก {len(table)} รีวิว")

        buffer = io.StringIO()
        table.to_csv(buffer, index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇️ ดาวน์โหลดผลลัพธ์ทั้งหมดเป็น CSV",
            buffer.getvalue().encode("utf-8-sig"),
            file_name="reviewlens_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ================================================================ แท็บ 3

with tab_doc:
    st.markdown(
        """
## ลำดับการทำงาน

```
ข้อความรีวิว
  │
  ├─ 1. Regex & Cleansing        ลบ URL / เบอร์โทร / LINE ID / อีเมล / HTML / อีโมจิ
  │                              แยกคำอังกฤษที่ติดกัน  →  clean_text, pii_removed
  │
  ├─ 2. Tokenization & Normalization
  │                              ยุบคำลากเสียง → ตัดคำ → แก้คำพิมพ์ผิด → ลบ stopwords
  │                              →  normalized_text, tokens, tokens_clean
  │
  ├─ 3. POS & NER                ติดป้ายชนิดคำ + สกัดชื่อเฉพาะ
  │                              →  pos_tags, brand, seller_shop, location, price, ...
  │
  ├─ 4. Topic Identification     จับ aspect 6 ด้าน + วิเคราะห์ขั้วราย aspect
  │                              →  aspects, aspect_sentiment, pros, cons
  │
  └─ 5. Screening                ใช้ PII ที่ลบไปเป็นสัญญาณจับสแปม
                                 →  credibility_score, review_status
```

## จุดที่ตัดสินใจออกแบบไว้เป็นพิเศษ

**ทำไมต้อง normalize ก่อน tokenize**
PyThaiNLP ตัด `"ส่งไวมากกกกก"` เป็น `['ส่ง','ไว','มาก','กก','กก']`
แต่ถ้ายุบคำลากเสียงก่อนจะได้ `['ส่ง','ไว','มาก']` ตามที่ควรเป็น

**ทำไมไม่ค้นคำแบบ substring ตรง ๆ**
ภาษาไทยไม่เว้นวรรค ทำให้ `"ทน"` (ทนทาน) ไปแมตช์กลางคำ `"ตัวแทน"`
และ `"เลย"` (จังหวัด) ไปแมตช์ใน `"ได้เลยค่ะ"`
ระบบจึงค้นบนข้อความที่ตัดคำแล้ว โดยบังคับให้แมตช์ต้องเริ่มที่ขอบเขตคำ

**ทำไมวัดขอบเขตคำปฏิเสธด้วยจำนวนคำ ไม่ใช่ระยะตัวอักษร**
ในประโยค `"ยังไม่มีปัญหาเลย ทนกว่าตัวเก่า"` คำว่า `"ไม่"` เป็นของ `"มีปัญหา"`
ไม่ใช่ของ `"ทน"` ที่อยู่ห่างออกไป 2 คำ ถ้าวัดด้วยระยะตัวอักษรจะพลิกขั้วผิด

**ทำไมใช้ทั้ง Regex และ NER**
spaCy มอง `"6195"` ในประโยค `"the inverter costs Rs. 6195"` เป็น `DATE` ไม่ใช่ `MONEY`
แต่ regex ราคาจับได้ถูก ส่วน NER เก่งกว่าตรงชื่อเฉพาะที่เขียนได้หลายแบบ

**ทำไมคำชมกลาง ๆ ไม่ถูกผูกกับ aspect ใด**
`"ประทับใจ"` ใช้ชมด้านไหนก็ได้ — `"ประทับใจร้านนี้"` คือบริการ ไม่ใช่คุณภาพสินค้า
ระบบจึงเก็บคำเหล่านี้ไว้ต่างหาก แล้วโยงเข้ากับ aspect ที่อยู่ใกล้ที่สุดในข้อความ

## Aspect ทั้ง 6 ด้าน
        """
    )
    st.dataframe(
        pd.DataFrame([
            {
                "ด้าน": f"{v['icon']} {v['label_th']}",
                "ตัวอย่างคำชม": ", ".join(v["pos"][:5]),
                "ตัวอย่างคำติ": ", ".join(v["neg"][:5]),
            }
            for v in ASPECTS.values()
        ]),
        use_container_width=True, hide_index=True,
    )
    st.markdown(
        """
## ชุดข้อมูลทดสอบ

| ไฟล์ | จำนวน | ที่มา |
|---|---|---|
| `data/reviews_flipkart_en.csv` | 500 | สุ่มแบบ stratified จาก [Flipkart Product Reviews](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset) (205,052 แถว) |
| `data/reviews_th.csv` | 44 | เขียนขึ้นเอง พร้อมกำกับ `expected_aspects` เป็น ground truth |
| `data/reviews_sample.csv` | 544 | รวมสองชุด — ไฟล์ที่แอปนี้โหลด |

ต้องเขียนชุดไทยเพิ่มเพราะชุด Flipkart ผ่านการ moderate มาแล้ว
**ไม่มี URL เลยสักแถว และมีเบอร์โทรเพียง 2 แถวจาก 205,052**
ทำให้ทดสอบโมดูลคัดกรองสแปมไม่ได้เลยถ้าใช้ชุดนั้นอย่างเดียว
        """
    )
