from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st

from app.database import list_recent_analyses
from app.inference import MODEL_OPTIONS, UPLOADS_ROOT, analyze_video, infer_label_from_name


st.set_page_config(
    page_title="Driver State Analyzer",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f3ea;
            --card: rgba(255, 252, 245, 0.88);
            --ink: #1f2937;
            --muted: #596579;
            --accent: #c26a2e;
            --accent-dark: #8b451a;
            --line: rgba(31, 41, 55, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(194, 106, 46, 0.16), transparent 28%),
                linear-gradient(180deg, #fbf8f1 0%, #f2ede4 100%);
            color: var(--ink);
        }
        .hero {
            padding: 1.4rem 1.5rem;
            border: 1px solid var(--line);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(255,248,235,0.96));
            box-shadow: 0 18px 40px rgba(87, 63, 34, 0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            color: #172033;
            letter-spacing: -0.02em;
        }
        .hero p {
            color: var(--muted);
            margin-top: 0.5rem;
            margin-bottom: 0;
            font-size: 1rem;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(80, 62, 36, 0.06);
        }
        .card h3 {
            margin-top: 0;
            margin-bottom: 0.35rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent-dark);
        }
        .big-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #111827;
            margin: 0;
        }
        .subtle {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.3rem;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 252, 245, 0.92), rgba(250, 243, 231, 0.98));
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 28px rgba(80, 62, 36, 0.06);
        }
        [data-testid="stMetricLabel"] {
            color: var(--accent-dark) !important;
            font-size: 0.9rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.02em;
        }
        [data-testid="stMetricLabel"] * {
            color: var(--accent-dark) !important;
        }
        [data-testid="stMetricValue"] {
            color: #172033 !important;
            font-weight: 800 !important;
            line-height: 1.1;
        }
        [data-testid="stMetricValue"] * {
            color: #172033 !important;
        }
        [data-testid="stMetricDelta"] {
            color: var(--muted) !important;
        }
        [data-testid="stMetricDelta"] * {
            color: var(--muted) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def file_bytes_for_download(path: str) -> bytes:
    return Path(path).read_bytes()


def sanitize_uploaded_filename(raw_name: str) -> str:
    base_name = Path(str(raw_name)).name.strip()
    if not base_name:
        return "uploaded_video.mp4"

    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", base_name)
    safe_name = safe_name.rstrip(". ")

    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }
    stem = Path(safe_name).stem.upper()
    if stem in reserved_names:
        safe_name = f"upload_{safe_name}"

    return safe_name or "uploaded_video.mp4"


def save_uploaded_video(uploaded_file) -> Path:
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_uploaded_filename(uploaded_file.name)
    target = UPLOADS_ROOT / safe_name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def render_overview_cards(summary: dict, quality: dict, timeline_df: pd.DataFrame) -> None:
    label_map = {
        "normal": "Normal",
        "drowsiness": "Uykulu",
        "distraction": "Dikkati Dagilmis",
    }
    label_text = label_map.get(summary["overall_label"], summary["overall_label"])

    usable_windows = int(len(timeline_df))
    total_duration = 0.0 if timeline_df.empty else float(timeline_df["window_end_time"].max())
    dominant_window_count = int(summary.get("dominant_window_count", 0))
    dominant_window_ratio = float(summary.get("dominant_window_ratio", 0.0))
    predicted_counts = summary.get("predicted_window_counts", {})
    distribution_parts = [
        f"{label_map.get(label, label)}: {int(predicted_counts.get(label, 0))}"
        for label in ("distraction", "drowsiness", "normal")
        if int(predicted_counts.get(label, 0)) > 0
    ]

    cols = st.columns(4)
    cards = [
        ("Genel Sonuc", label_text, f"{usable_windows} pencere, yaklasik sure: {total_duration:.1f} sn"),
        ("Model Olasiligi", f"%{summary['overall_confidence'] * 100:.1f}", "Baskin sinifin ortalama olasiligi"),
        (
            "Pencere Baskinligi",
            f"%{dominant_window_ratio * 100:.1f}",
            f"{dominant_window_count} / {usable_windows} pencere bu sinifa gitti",
        ),
        ("Risk Skoru", f"{summary['risk_score']:.1f}/100", "Unsafe olasiliklarin toplami"),
    ]

    for col, (title, value, subtitle) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{title}</h3>
                    <p class="big-value">{value}</p>
                    <p class="subtle">{subtitle}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if distribution_parts:
        st.caption("Pencere dagilimi: " + " | ".join(distribution_parts))

    st.write("")
    quality_cols = st.columns(4)
    quality_cards = [
        ("Yuz Tespit Orani", f"%{quality['detected_face_ratio'] * 100:.1f}"),
        ("Gecerli Poz Orani", f"%{quality['valid_pose_ratio'] * 100:.1f}"),
        ("Gecerli EAR Orani", f"%{quality['valid_ear_ratio'] * 100:.1f}"),
        ("Supheli EAR Orani", f"%{quality['suspicious_ear_ratio'] * 100:.1f}"),
    ]
    for col, (title, value) in zip(quality_cols, quality_cards):
        with col:
            st.metric(title, value)


def render_probability_summary(summary: dict) -> None:
    probability_df = pd.DataFrame(
        {
            "Sinif": ["Normal", "Uykulu", "Dikkati Dagilmis"],
            "Ortalama Olasilik": [
                summary["mean_probabilities"].get("normal", 0.0),
                summary["mean_probabilities"].get("drowsiness", 0.0),
                summary["mean_probabilities"].get("distraction", 0.0),
            ],
        }
    )
    st.bar_chart(probability_df.set_index("Sinif"))


def render_timeline(timeline_df: pd.DataFrame) -> None:
    chart_df = timeline_df[
        ["window_end_time", "prob_normal", "prob_drowsiness", "prob_distraction", "window_risk_score"]
    ].copy()
    chart_df = chart_df.rename(
        columns={
            "window_end_time": "Zaman",
            "prob_normal": "Normal",
            "prob_drowsiness": "Uykulu",
            "prob_distraction": "Dikkati Dagilmis",
            "window_risk_score": "Risk Skoru",
        }
    )
    st.line_chart(chart_df.set_index("Zaman"))

    table_df = timeline_df[
        [
            "window_id",
            "window_start_time",
            "window_end_time",
            "predicted_label",
            "prob_normal",
            "prob_drowsiness",
            "prob_distraction",
            "window_risk_score",
            "perclos_percent",
            "mean_ear",
            "max_abs_yaw",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "window_id": "Pencere",
            "window_start_time": "Baslangic",
            "window_end_time": "Bitis",
            "predicted_label": "Tahmin",
            "prob_normal": "Normal Olasilik",
            "prob_drowsiness": "Uykulu Olasilik",
            "prob_distraction": "Dikkati Dagilmis Olasilik",
            "window_risk_score": "Risk",
            "perclos_percent": "PERCLOS %",
            "mean_ear": "Ort. EAR",
            "max_abs_yaw": "Maks. |Yaw|",
        }
    )
    st.dataframe(table_df, use_container_width=True)


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Driver State Analyzer</h1>
            <p>Surucu videosunu yukle, mevcut bitirme projesi pipeline'ini kullanarak pencere bazli analiz al ve risk akisini gor.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Ayarlar")
        model_key = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index("xgboost_gaze_high_confidence"),
            format_func=lambda key: MODEL_OPTIONS[key]["label"],
            help="Varsayilan secim repodaki en iyi dogrulanmis modeldir.",
        )
        fast_mode = st.checkbox(
            "Hizli mod",
            value=True,
            help="Daha hizli demo icin daha hafif landmark isleme kullanir.",
        )
        label_override = st.selectbox(
            "Kaynak Etiketi",
            options=["auto", "normal", "drowsiness", "distraction", "uploaded"],
            index=0,
            help="Veri kumesinden gelen videolarda dosya adindan otomatik tahmin edilir; gerekirse elle degistirebilirsin.",
        )
        st.caption(MODEL_OPTIONS[model_key]["description"])
        st.info(
            "Urun inference katmani artik tek MediaPipe gecisinde hem baseline hem gaze ozelliklerini cikariyor. "
            "Gaze destekli modeller hala daha agir, ama onceki iki ayri video gecisine gore daha verimli."
        )

    uploaded_file = st.file_uploader(
        "Surucu videosu yukle",
        type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"],
    )

    if not uploaded_file:
        st.stop()

    video_path = save_uploaded_video(uploaded_file)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.video(str(video_path))
    with right:
        st.markdown("**Dosya**")
        st.write(uploaded_file.name)
        st.markdown("**Algilanan Etiket**")
        detected_label = infer_label_from_name(uploaded_file.name)
        st.write(detected_label)
        st.markdown("**Boyut**")
        st.write(f"{uploaded_file.size / (1024 * 1024):.2f} MB")
        st.markdown("**Hazir durum**")
        st.write("Analize basmak icin asagidaki butonu kullan.")

    if not st.button("Videoyu Analiz Et", type="primary", use_container_width=True):
        st.stop()

    with st.spinner("Video isleniyor, ozellikler cikariliyor ve model tahmini uretiliyor..."):
        result = analyze_video(
            video_path=video_path,
            model_key=model_key,
            fast_mode=fast_mode,
            label_override=None if label_override == "auto" else label_override,
        )

    if result["status"] != "ok":
        st.error(result["message"])
        quality = result.get("quality", {})
        if quality:
            st.json(quality)
        st.stop()

    timeline_df = pd.DataFrame(result["timeline"])
    summary = result["summary"]
    quality = result["quality"]

    render_overview_cards(summary, quality, timeline_df)
    st.caption(f"Kaynak etiketi: `{result['source_label']}` | Modalite: `{result['source_modality']}`")
    st.success(
        f"Analiz veritabanina kaydedildi. Kayit No: `{result['analysis_id']}` | DB: `{result['database_path']}`"
    )

    st.write("")
    chart_col, summary_col = st.columns([1.35, 0.65])
    with chart_col:
        st.subheader("Zaman Cizelgesi")
        render_timeline(timeline_df)
    with summary_col:
        st.subheader("Ortalama Olasiliklar")
        render_probability_summary(summary)
        st.subheader("Cikti Dosyalari")
        for label, artifact_path in result["artifacts"].items():
            st.caption(f"{label}: {artifact_path}")
        st.subheader("Son Analizler")
        recent_df = pd.DataFrame(list_recent_analyses(limit=5))
        if not recent_df.empty:
            st.dataframe(recent_df, use_container_width=True, hide_index=True)

    downloads_col1, downloads_col2 = st.columns(2)
    with downloads_col1:
        predictions_path = result["artifacts"]["window_predictions_csv"]
        st.download_button(
            "Pencere Tahminlerini Indir",
            data=file_bytes_for_download(predictions_path),
            file_name=Path(predictions_path).name,
            mime="text/csv",
            use_container_width=True,
        )
    with downloads_col2:
        frame_path = result["artifacts"]["frame_csv"]
        st.download_button(
            "Ham Frame Ozelliklerini Indir",
            data=file_bytes_for_download(frame_path),
            file_name=Path(frame_path).name,
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("MVP Notlari"):
        st.write(
            "Bu surum dogrudan repo icindeki mevcut feature extraction, gaze extraction ve egitilmis model paketlerini kullanir. "
            "Canli kamera, sesli alarm ve daha hizli arka plan isleme bir sonraki iterasyonda eklenebilir."
        )


if __name__ == "__main__":
    main()
