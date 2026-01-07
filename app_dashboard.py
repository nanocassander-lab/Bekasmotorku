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
st.caption("Analisis harga dan karakteristik motor bekas")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("data_bersih.csv")

df = load_data()

# =========================================================
# SIDEBAR FILTER
# =========================================================
st.sidebar.header("🔍 Filter Data")

company = st.sidebar.multiselect(
    "Pilih Brand",
    options=df["company"].unique(),
    default=df["company"].unique()
)

location = st.sidebar.multiselect(
    "Pilih Lokasi",
    options=df["Location"].unique(),
    default=df["Location"].unique()
)

year_min, year_max = st.sidebar.slider(
    "Rentang Tahun",
    int(df["year"].min()),
    int(df["year"].max()),
    (int(df["year"].min()), int(df["year"].max()))
)

price_min, price_max = st.sidebar.slider(
    "Rentang Harga",
    int(df["price"].min()),
    int(df["price"].max()),
    (int(df["price"].min()), int(df["price"].max()))
)

travel_min, travel_max = st.sidebar.slider(
    "Rentang Kilometer",
    int(df["travelled"].min()),
    int(df["travelled"].max()),
    (int(df["travelled"].min()), int(df["travelled"].max()))
)

# =========================================================
# APPLY FILTER
# =========================================================
df_f = df[
    (df["company"].isin(company)) &
    (df["Location"].isin(location)) &
    (df["year"].between(year_min, year_max)) &
    (df["price"].between(price_min, price_max)) &
    (df["travelled"].between(travel_min, travel_max))
]

# =========================================================
# KPI
# =========================================================
k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Motor", df_f.shape[0])
k2.metric("Median Harga", f"{df_f['price'].median():,.0f}")
k3.metric("Rata-rata KM", f"{df_f['travelled'].mean():,.0f}")
k4.metric("Rentang Tahun", f"{df_f['year'].min()} - {df_f['year'].max()}")

st.divider()

# =========================================================
# VISUALISASI
# =========================================================
c1, c2 = st.columns(2)

with c1:
    fig_box = px.box(
        df_f,
        y="price",
        title="Boxplot Harga Motor Bekas"
    )
    st.plotly_chart(fig_box, use_container_width=True)

with c2:
    fig_hist = px.histogram(
        df_f,
        x="price",
        nbins=30,
        title="Distribusi Harga Motor Bekas"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("🔎 Hubungan Harga dan Kilometer")
fig_scatter = px.scatter(
    df_f,
    x="travelled",
    y="price",
    color="company",
    hover_data=["name", "Location", "year"],
    title="Harga vs Kilometer"
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("📈 Rata-rata Harga per Tahun")
avg_year = df_f.groupby("year", as_index=False)["price"].mean()
fig_line = px.line(
    avg_year,
    x="year",
    y="price",
    markers=True,
    title="Tren Harga Motor Bekas"
)
st.plotly_chart(fig_line, use_container_width=True)

# =========================================================
# DATA TABLE
# =========================================================
st.subheader("📋 Data Motor Bekas")
st.dataframe(df_f, use_container_width=True, hide_index=True)

st.caption("Dashboard dibuat menggunakan Streamlit & Plotly")
