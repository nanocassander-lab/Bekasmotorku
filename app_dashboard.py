import streamlit as st
import pandas as pd
import plotly.express as px

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Motor Bekas",
    page_icon="🏍️",
    layout="wide"
)

st.title("🏍️ Dashboard Analisis Motor Bekas")
st.caption("Dashboard interaktif untuk eksplorasi data motor bekas (filter, visualisasi, dan statistik).")

# =========================================================
# LOAD DATA
# =========================================================
DEFAULT_LOCAL_PATH = "data_bersih.csv"
DEFAULT_URL = "https://raw.githubusercontent.com/nanocassander-lab/Bekasmotorku/main/motor_second.csv"

with st.sidebar:
    st.header("⚙️ Sumber Data")
    source = st.radio(
        "Pilih sumber data",
        ["CSV lokal (data_bersih.csv)", "Upload CSV", "URL (GitHub/raw)"],
        index=0
    )

@st.cache_data(show_spinner=False)
def load_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url)

df = None

if source == "CSV lokal (data_bersih.csv)":
    try:
        df = load_from_path(DEFAULT_LOCAL_PATH)
    except Exception as e:
        st.sidebar.error(f"Gagal membaca {DEFAULT_LOCAL_PATH}. Coba opsi Upload/URL.\n\nDetail: {e}")
        df = None

elif source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

else:
    url = st.sidebar.text_input("Masukkan URL CSV (raw)", value=DEFAULT_URL)
    try:
        df = load_from_url(url)
    except Exception as e:
        st.sidebar.error(f"Gagal membaca URL.\n\nDetail: {e}")
        df = None

if df is None or df.empty:
    st.stop()

# =========================================================
# PRE-PROCESSING RINGAN (AMAN)
# =========================================================
# Rapikan nama kolom: hilangkan spasi di tepi
df.columns = [c.strip() for c in df.columns]

# Konversi angka yang mungkin terbaca sebagai object (jika ada)
for col in df.columns:
    if df[col].dtype == "object":
        # coba ubah kolom object yang sebenarnya angka (contoh: "12,000" / "12000")
        s = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        numeric = pd.to_numeric(s, errors="coerce")
        # kalau mayoritas bisa jadi angka, gunakan numeric
        if numeric.notna().mean() >= 0.8:
            df[col] = numeric

# Identifikasi kolom numerik & kategorikal
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

# =========================================================
# SIDEBAR: FILTER
# =========================================================
with st.sidebar:
    st.header("🔍 Filter Data")
    st.metric("Total Baris (awal)", int(df.shape[0]))

df_f = df.copy()

# Filter kategorikal (multi-kolom)
if cat_cols:
    with st.sidebar.expander("Filter Kategori", expanded=True):
        for c in cat_cols:
            opts = sorted([x for x in df[c].dropna().unique().tolist()])
            if len(opts) == 0:
                continue
            selected = st.multiselect(f"{c}", options=opts, default=opts)
            df_f = df_f[df_f[c].isin(selected)]

# Filter numerik (slider per kolom utama yang umum dipakai)
common_numeric_priority = [c for c in ["tahun", "harga", "odometer", "pajak", "konsumsiBBM", "mesin"] if c in num_cols]
other_numeric = [c for c in num_cols if c not in common_numeric_priority]
numeric_for_filter = common_numeric_priority + other_numeric

if numeric_for_filter:
    with st.sidebar.expander("Filter Numerik", expanded=True):
        for c in numeric_for_filter:
            series = df[c].dropna()
            if series.empty:
                continue
            is_int = pd.api.types.is_integer_dtype(df[c])
            if is_int:
                min_val = int(series.min())
                max_val = int(series.max())
                step = 1
                lo, hi = st.slider(
                    f"Rentang {c}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=step
                )
            else:
                min_val = float(series.min())
                max_val = float(series.max())
                span = max_val - min_val
                step = span / 100.0 if span > 0 else 0.1
                lo, hi = st.slider(
                    f"Rentang {c}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=step,
                    format="%.2f"
                )

            df_f = df_f[(df_f[c] >= lo) & (df_f[c] <= hi)]

with st.sidebar:
    st.metric("Total Baris (setelah filter)", int(df_f.shape[0]))
    st.divider()

# =========================================================
# KPI RINGKASAN
# =========================================================
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

def safe_metric(series: pd.Series, fn: str):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    if fn == "mean":
        return float(s.mean())
    if fn == "median":
        return float(s.median())
    if fn == "min":
        return float(s.min())
    if fn == "max":
        return float(s.max())
    return None

with kpi1:
    st.metric("Jumlah Data", int(df_f.shape[0]))

with kpi2:
    if "harga" in df_f.columns:
        m = safe_metric(df_f["harga"], "median")
        st.metric("Median Harga", f"{m:,.0f}" if m is not None else "-")
    else:
        st.metric("Median Harga", "-")

with kpi3:
    if "odometer" in df_f.columns:
        m = safe_metric(df_f["odometer"], "mean")
        st.metric("Rata-rata Odometer", f"{m:,.0f}" if m is not None else "-")
    else:
        st.metric("Rata-rata Odometer", "-")

with kpi4:
    if "tahun" in df_f.columns:
        lo = safe_metric(df_f["tahun"], "min")
        hi = safe_metric(df_f["tahun"], "max")
        if lo is not None and hi is not None:
            st.metric("Rentang Tahun", f"{int(lo)}–{int(hi)}")
        else:
            st.metric("Rentang Tahun", "-")
    else:
        st.metric("Rentang Tahun", "-")

st.divider()

# =========================================================
# TABS: DATA, VISUAL, STATISTIK
# =========================================================
tab_data, tab_viz, tab_stats = st.tabs(["📋 Data", "📊 Visualisasi", "📈 Statistik"])

with tab_data:
    st.subheader("📋 Data Setelah Filter")
    st.dataframe(df_f, use_container_width=True, hide_index=True)

    # Download hasil filter
    csv_bytes = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV (hasil filter)",
        data=csv_bytes,
        file_name="data_motor_filtered.csv",
        mime="text/csv"
    )

with tab_viz:
    st.subheader("📊 Visualisasi Utama")

    # Kontrol visual
    viz_cols = st.columns(3)
    with viz_cols[0]:
        x_num = st.selectbox("Pilih variabel numerik (X)", options=num_cols if num_cols else df_f.columns.tolist())
    with viz_cols[1]:
        y_num = st.selectbox("Pilih variabel numerik (Y)", options=num_cols if num_cols else df_f.columns.tolist(), index=1 if len(num_cols) > 1 else 0)
    with viz_cols[2]:
        color_by = st.selectbox("Warna berdasarkan (opsional)", options=["(none)"] + cat_cols)

    # Histogram X
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df_f,
            x=x_num,
            nbins=30,
            title=f"Distribusi {x_num}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Box plot Y
    with c2:
        fig_box = px.box(
            df_f,
            y=y_num,
            title=f"Boxplot {y_num}"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Scatter X vs Y
    st.subheader("🔎 Hubungan Antar Variabel")
    if color_by != "(none)":
        fig_scatter = px.scatter(
            df_f,
            x=x_num,
            y=y_num,
            color=color_by,
            hover_data=df_f.columns,
            title=f"{y_num} vs {x_num} (warna: {color_by})"
        )
    else:
        fig_scatter = px.scatter(
            df_f,
            x=x_num,
            y=y_num,
            hover_data=df_f.columns,
            title=f"{y_num} vs {x_num}"
        )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Grafik khusus jika kolom umum tersedia
    st.subheader("📌 Grafik Ringkas (Jika Kolom Tersedia)")
    g1, g2 = st.columns(2)

    if {"tahun", "harga"}.issubset(df_f.columns):
        with g1:
            tmp = df_f.groupby("tahun", as_index=False)["harga"].mean().sort_values("tahun")
            fig_line = px.line(tmp, x="tahun", y="harga", markers=True, title="Rata-rata Harga per Tahun")
            st.plotly_chart(fig_line, use_container_width=True)

    if {"jenis", "harga"}.issubset(df_f.columns):
        with g2:
            fig_box2 = px.box(df_f, x="jenis", y="harga", title="Harga per Jenis")
            st.plotly_chart(fig_box2, use_container_width=True)

    if {"model"}.issubset(df_f.columns):
        top_n = st.slider("Top N Model (berdasarkan jumlah data)", 5, 30, 10)
        counts = df_f["model"].value_counts().head(top_n).reset_index()
        counts.columns = ["model", "jumlah"]
        fig_bar = px.bar(counts, x="model", y="jumlah", title=f"Top {top_n} Model (Jumlah Data)")
        st.plotly_chart(fig_bar, use_container_width=True)

with tab_stats:
    st.subheader("📈 Statistik Deskriptif")
    if num_cols:
        st.write(df_f[num_cols].describe().T)
    else:
        st.info("Tidak ada kolom numerik terdeteksi.")

    st.subheader("🧾 Info Kolom & Missing Values")
    info = pd.DataFrame({
        "kolom": df_f.columns,
        "dtype": [str(df_f[c].dtype) for c in df_f.columns],
        "missing": [int(df_f[c].isna().sum()) for c in df_f.columns],
        "unique": [int(df_f[c].nunique(dropna=True)) for c in df_f.columns],
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Dibuat dengan Streamlit + Plotly • Dashboard Motor Bekas")