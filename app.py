import os, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

# ─── SAYFA AYARI ───────────────────────────────────────────
st.set_page_config(
    page_title="Selvese Pusulası",
    page_icon="🧭",
    layout="wide"
)

# ─── STİL ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1c2333;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
        border-left: 4px solid #4a9eff;
    }
    .sat { border-left-color: #ff4b4b !important; }
    .bekle { border-left-color: #ffa500 !important; }
    .hazirlan { border-left-color: #00cc88 !important; }
    .rapor-box {
        background: #1c2333;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 16px;
        line-height: 1.8;
    }
    h1 { color: #ffffff; }
    .stButton>button {
        background: #4a9eff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 32px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { background: #2d7dd2; }
</style>
""", unsafe_allow_html=True)

# ─── YARDIMCI FONKSİYONLAR ─────────────────────────────────
def clamp(x, lo, hi): return max(lo, min(hi, x))
def safe_float(x):
    try: return float(x)
    except: return None
def pct(a, b):
    if a and b and a != 0: return (b/a - 1) * 100
    return None
def ema(s, span): return s.ewm(span=span, adjust=False).mean()
def rsi(close, period=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    rs = up.rolling(period).mean() / dn.rolling(period).mean().replace(0, pd.NA)
    return 100 - (100 / (1 + rs))
def macd_hist(close):
    m = ema(close, 12) - ema(close, 26)
    s = ema(m, 9)
    return m - s

# ─── VERİ ÇEKİCİLER ────────────────────────────────────────
@st.cache_data(ttl=300)
def get_yahoo(ticker, interval, period):
    try:
        df = yf.download(ticker, interval=interval, period=period,
                         auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.dropna()
    except: return None

@st.cache_data(ttl=3600)
def get_us2y():
    try:
        r = requests.get(
            "https://home.treasury.gov/resource-center/data-chart-center/"
            "interest-rates/pages/xmlview?data=daily_treasury_yield_curve",
            timeout=15)
        import re
        m = re.findall(r"BC_2YEAR[^0-9]*([\d\.]+)", r.text)
        if m: return float(m[-1])
    except: pass
    return None

@st.cache_data(ttl=3600)
def get_de2y():
    try:
        url = ("https://api.statistiken.bundesbank.de/rest/data/"
               "BBSSY/D.REN.EUR.A610.000000WT0202.A")
        r = requests.get(url, params={"format": "sdmx_csv"}, timeout=15)
        if r.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            col = [c for c in df.columns if "OBS" in c.upper()]
            if col:
                v = pd.to_numeric(df[col[0]], errors="coerce").dropna()
                if len(v): return float(v.iloc[-1])
    except: pass
    return None

# ─── SKORLAMA ──────────────────────────────────────────────
def score_dxy(dxy_pct):
    if dxy_pct is None: return 50, "Veri yok"
    s = clamp(50 - (dxy_pct / 1.5) * 25, 0, 100)
    if dxy_pct > 0.5: yorum = f"DXY güçleniyor (+{dxy_pct:.1f}%) → EUR baskı altında"
    elif dxy_pct < -0.5: yorum = f"DXY zayıflıyor ({dxy_pct:.1f}%) → EUR için pozitif"
    else: yorum = f"DXY nötr ({dxy_pct:.1f}%)"
    return s, yorum

def score_rates(us, de):
    if us is None or de is None: return 50, "Faiz verisi eksik"
    spread = us - de
    s = clamp(60 - (spread / 3) * 30, 0, 100)
    yorum = f"ABD-Almanya 2Y spread: {spread:.2f}% → {'EUR aleyhine' if spread > 1.5 else 'EUR lehine' if spread < 0.5 else 'Nötr'}"
    return s, yorum

def score_vix(vix):
    if vix is None: return 50, "VIX verisi yok"
    s = clamp(70 - (vix - 12) * 2, 0, 100)
    yorum = f"VIX: {vix:.1f} → {'Risk iştahı yüksek' if vix < 18 else 'Risk iştahı düşük' if vix > 25 else 'Normal volatilite'}"
    return s, yorum

def score_technical(df):
    if df is None or len(df) < 60: return 50, "Teknik veri yetersiz"
    ma20 = df["Close"].rolling(20).mean().iloc[-1]
    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    r = rsi(df["Close"]).iloc[-1]
    h = macd_hist(df["Close"]).iloc[-1]
    s = clamp(
        0.45*(65 if ma20>ma50 else 35) +
        0.30*(50+(r-50)*1.2) +
        0.25*(50+h*2000),
        0, 100)
    trend = "Yukarı trend" if ma20 > ma50 else "Aşağı trend"
    rsi_yorum = "Aşırı alım" if r > 70 else "Aşırı satım" if r < 30 else "Normal"
    yorum = f"MA20/50: {trend} | RSI: {r:.0f} ({rsi_yorum}) | MACD Hist: {h:.5f}"
    return s, yorum

def score_form(spot, support, resistance):
    if None in (spot, support, resistance) or resistance == support:
        return 50, "Formasyon hesaplanamadı"
    pos = (spot - support) / (resistance - support)
    s = clamp(50 + (pos - 0.5) * 20, 0, 100)
    yorum = f"Fiyat aralık içinde %{pos*100:.0f} konumunda | Destek: {support:.4f} | Direnç: {resistance:.4f}"
    return s, yorum

# ─── ANA HESAPLAMA ──────────────────────────────────────────
def hesapla():
    with st.spinner("Veriler çekiliyor..."):
        eur_1d = get_yahoo("EURUSD=X", "1d", "600d")
        eur_4h = get_yahoo("EURUSD=X", "4h", "180d")
        dxy_df = get_yahoo("DX-Y.NYB", "1d", "90d")
        vix_df = get_yahoo("^VIX", "1d", "30d")
        us2y   = get_us2y()
        de2y   = get_de2y()

    spot = safe_float(eur_1d["Close"].iloc[-1]) if eur_1d is not None else None
    support = float(eur_1d["Low"].tail(60).min()) if eur_1d is not None else None
    resistance = float(eur_1d["High"].tail(60).max()) if eur_1d is not None else None

    dxy_pct = None
    if dxy_df is not None and len(dxy_df) >= 4:
        dxy_pct = pct(float(dxy_df["Close"].iloc[-4]), float(dxy_df["Close"].iloc[-1]))

    vix_val = safe_float(vix_df["Close"].iloc[-1]) if vix_df is not None else None

    s_dxy,  y_dxy  = score_dxy(dxy_pct)
    s_faiz, y_faiz = score_rates(us2y, de2y)
    s_risk, y_risk = score_vix(vix_val)
    s_tek,  y_tek  = score_technical(eur_1d)
    s_form, y_form = score_form(spot, support, resistance)

    weights = {"DXY":0.25, "Faiz":0.20, "Risk":0.15, "Teknik":0.25, "Form":0.15}
    scores  = {"DXY":s_dxy, "Faiz":s_faiz, "Risk":s_risk, "Teknik":s_tek, "Form":s_form}
    ede = round(sum(scores[k]*weights[k] for k in scores), 1)

    # Karar
    if ede >= 65:
        karar, renk, emoji = "SAT", "sat", "🔴"
    elif ede >= 52:
        karar, renk, emoji = "HAZIRLAN", "hazirlan", "🟡"
    else:
        karar, renk, emoji = "BEKLE", "bekle", "🟢"

    return {
        "spot": spot, "support": support, "resistance": resistance,
        "ede": ede, "karar": karar, "renk": renk, "emoji": emoji,
        "us2y": us2y, "de2y": de2y, "dxy_pct": dxy_pct, "vix": vix_val,
        "scores": scores, "yorumlar": {
            "DXY": y_dxy, "Faiz": y_faiz, "Risk": y_risk,
            "Teknik": y_tek, "Form": y_form
        },
        "eur_1d": eur_1d, "eur_4h": eur_4h,
        "zaman": datetime.now().strftime("%d.%m.%Y %H:%M")
    }

# ─── BAROMETRE ─────────────────────────────────────────────
def barometre(ede, karar, renk):
    renk_map = {"sat": "#ff4b4b", "hazirlan": "#ffa500", "bekle": "#00cc88"}
    c = renk_map[renk]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ede,
        number={"font": {"size": 48, "color": c}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
            "bar": {"color": c, "thickness": 0.25},
            "bgcolor": "#1c2333",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "#0d2137"},
                {"range": [50, 65], "color": "#1a2e1a"},
                {"range": [65, 100],"color": "#2e1a1a"},
            ],
            "threshold": {
                "line": {"color": c, "width": 4},
                "thickness": 0.75,
                "value": ede
            }
        },
        title={"text": f"<b>{karar}</b>", "font": {"size": 28, "color": c}}
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117",
        font={"color": "#ffffff"},
        height=280,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    return fig

# ─── RAPOR METNİ ───────────────────────────────────────────
def rapor_metni(d):
    karar_aciklama = {
        "SAT": "EDE skoru yüksek bölgede. Teknik ve makro göstergeler EUR için elverişsiz. Satış penceresi açık görünüyor.",
        "HAZIRLAN": "EDE skoru orta bölgede. Piyasa yön arayışında. Pozisyon izlenmeli, satış için net sinyal beklenmeli.",
        "BEKLE": "EDE skoru düşük bölgede. EUR için güç göstergeleri baskın. Satış için daha iyi fırsat beklenebilir."
    }

    spread = None
    if d["us2y"] and d["de2y"]:
        spread = d["us2y"] - d["de2y"]

    metin = f"""
**📅 {d['zaman']} | Selvese EUR Satış Pusulası**

---

**📍 Piyasa Durumu**
- EUR/USD Spot: **{d['spot']:.4f}** (60 günlük aralık: {d['support']:.4f} – {d['resistance']:.4f})
- DXY 3 günlük değişim: **{f"{d['dxy_pct']:+.2f}%" if d['dxy_pct'] else 'Hesaplanamadı'}**
- VIX: **{f"{d['vix']:.1f}" if d['vix'] else 'N/A'}**
- ABD 2Y: **{f"{d['us2y']:.2f}%" if d['us2y'] else 'N/A'}** | Almanya 2Y: **{f"{d['de2y']:.2f}%" if d['de2y'] else 'N/A'}** | Spread: **{f"{spread:.2f}%" if spread else 'N/A'}**

---

**📊 Skor Detayları**
| Kategori | Skor | Ağırlık | Yorum |
|----------|------|---------|-------|
| DXY | {d['scores']['DXY']:.0f} | %25 | {d['yorumlar']['DXY']} |
| Faiz | {d['scores']['Faiz']:.0f} | %20 | {d['yorumlar']['Faiz']} |
| Risk | {d['scores']['Risk']:.0f} | %15 | {d['yorumlar']['Risk']} |
| Teknik | {d['scores']['Teknik']:.0f} | %25 | {d['yorumlar']['Teknik']} |
| Formasyon | {d['scores']['Form']:.0f} | %15 | {d['yorumlar']['Form']} |

---

**🧭 EDE Skoru: {d['ede']} / 100 → {d['emoji']} {d['karar']}**

{karar_aciklama[d['karar']]}

---

**📌 Melih'in Karar Alanı**
*(Kendi değerlendirmeni buraya ekleyebilirsin)*

---

**💭 Eğer işlemi ben yapsaydım**
{"EDE 65 üzerinde. Kademeli satış başlatırdım — tümünü değil, pozisyonun %30-50'sini." if d['karar']=='SAT' else "EDE 52-65 arasında. Bekler, bir sonraki raporu takip ederdim. Acelesiz." if d['karar']=='HAZIRLAN' else "EDE 52 altında. Hiç satmaz, EUR'da kalırdım. Daha iyi fırsat yakın değil."}

---
*Herkesi dinlerim ama kararı ben veririm. Bu rapor bilgi amaçlıdır, yatırım tavsiyesi değildir.*
"""
    return metin

# ─── ARAYÜZ ────────────────────────────────────────────────
st.title("🧭 Selvese EUR Satış Pusulası")
st.caption("Kapanışta veri, sabah karar.")

col_btn, col_zaman = st.columns([2, 3])
with col_btn:
    guncelle = st.button("🔄 Verileri Güncelle")

if "data" not in st.session_state:
    st.session_state.data = None

if guncelle:
    st.session_state.data = hesapla()

if st.session_state.data is None:
    st.info("👆 'Verileri Güncelle' butonuna basarak analizi başlat.")
    st.stop()

d = st.session_state.data

with col_zaman:
    st.caption(f"Son güncelleme: {d['zaman']}")

# ─── ANA LAYOUT ────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.plotly_chart(barometre(d["ede"], d["karar"], d["renk"]),
                    use_container_width=True)

    st.markdown(f"""
    <div class="metric-card">
        <small>EUR/USD Spot</small><br>
        <b style="font-size:22px">{d['spot']:.4f}</b>
    </div>
    <div class="metric-card">
        <small>EDE Skoru</small><br>
        <b style="font-size:22px">{d['ede']} / 100</b>
    </div>
    <div class="metric-card">
        <small>ABD–DE 2Y Spread</small><br>
        <b style="font-size:22px">{f"{d['us2y']-d['de2y']:.2f}%" if d['us2y'] and d['de2y'] else 'N/A'}</b>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Skor barları
    st.markdown("#### 📊 Kategori Skorları")
    for k, v in d["scores"].items():
        renk = "#ff4b4b" if v > 65 else "#00cc88" if v < 45 else "#ffa500"
        st.markdown(f"""
        <div style="margin:6px 0">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="color:#ccc">{k}</span>
                <span style="color:{renk};font-weight:600">{v:.0f}</span>
            </div>
            <div style="background:#1c2333;border-radius:4px;height:8px">
                <div style="background:{renk};width:{v}%;height:8px;border-radius:4px"></div>
            </div>
            <small style="color:#666">{d['yorumlar'][k]}</small>
        </div>
        """, unsafe_allow_html=True)

# ─── RAPOR ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📋 Analiz Raporu")
with st.container():
    st.markdown(f'<div class="rapor-box">', unsafe_allow_html=True)
    st.markdown(rapor_metni(d))
    st.markdown('</div>', unsafe_allow_html=True)

# ─── GRAFİK ────────────────────────────────────────────────
if d["eur_1d"] is not None:
    st.markdown("---")
    st.markdown("#### 📈 EUR/USD Günlük Grafik (Son 90 gün)")
    df_plot = d["eur_1d"].tail(90)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["Open"], high=df_plot["High"],
        low=df_plot["Low"],  close=df_plot["Close"],
        name="EUR/USD",
        increasing_line_color="#00cc88",
        decreasing_line_color="#ff4b4b"
    ))
    fig.add_hline(y=d["support"], line_dash="dot",
                  line_color="#ffa500", annotation_text="Destek")
    fig.add_hline(y=d["resistance"], line_dash="dot",
                  line_color="#4a9eff", annotation_text="Direnç")
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1c2333",
        font={"color":"#ffffff"},
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Selvese Pusulası • Bu uygulama bilgi amaçlıdır, yatırım tavsiyesi içermez.")
