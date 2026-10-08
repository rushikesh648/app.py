import io
import urllib.request
import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import streamlit as st
from adjustText import adjust_text
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

st.set_page_config(
    page_title="NFL Analytics Dashboard", layout="wide", page_icon="🏈"
)

st.title("🏈 NFL Advanced Passing & Situational Analytics")
st.markdown(
    "Explore **Expected Points Added (EPA)**, **Completion Percentage Over Expectation (CPOE)**, and situational 3rd-down dynamics."
)


# --- Data Caching ---
@st.cache_data(show_spinner="Loading play-by-play data from nflverse...")
def load_data(season):
    pbp = nfl.import_pbp_data([season])
    teams = nfl.import_team_desc()[
        ["team_abbr", "team_logo_espn", "team_color", "team_color2"]
    ]
    return pbp, teams


# --- Sidebar Filters ---
st.sidebar.header("Controls & Filters")
selected_season = st.sidebar.selectbox("Season", [2024, 2023, 2022], index=0)
min_attempts = st.sidebar.slider(
    "Min. Pass Attempts (CPOE Plot)",
    min_value=50,
    max_value=400,
    value=150,
    step=25,
)
filter_garbage = st.sidebar.checkbox(
    "Filter Garbage Time (5% <= WP <= 95%)", value=True
)

pbp, teams = load_data(selected_season)

# Apply situational cleaning
base_filter = (
    (pbp["play_type"].isin(["pass", "run"]))
    & (pbp["epa"].notna())
    & (pbp["qb_kneel"] == 0)
    & (pbp["qb_spike"] == 0)
    & (pbp["two_point_attempt"] == 0)
)
if filter_garbage:
    base_filter = base_filter & (pbp["wp"].between(0.05, 0.95))

clean_pbp = pbp[base_filter].copy()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(
    [
        "QB EPA vs. CPOE",
        "Early vs. Late Down Team Splits",
        "3rd Down Scramble Conversions",
    ]
)

# -------------------------------------------------------------
# TAB 1: EPA vs CPOE Scatter Plot
# -------------------------------------------------------------
with tab1:
    st.subheader("Quarterback Accuracy & Efficiency Matrix")

    pass_plays = clean_pbp[
        (clean_pbp["pass_attempt"] == 1)
        & (clean_pbp["cpoe"].notna())
        & (clean_pbp["passer_player_name"].notna())
    ]

    qb_summary = (
        pass_plays.groupby(["passer_player_name", "posteam"])
        .agg(
            attempts=("cpoe", "count"),
            avg_cpoe=("cpoe", "mean"),
            epa_per_pass=("epa", "mean"),
        )
        .reset_index()
    )

    qualified_qbs = qb_summary[qb_summary["attempts"] >= min_attempts].merge(
        teams, left_on="posteam", right_on="team_abbr", how="left"
    )

    # Cache logos
    logo_cache = {}

    def fetch_logo(url):
        if not url or pd.isna(url):
            return None
        if url not in logo_cache:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    logo_cache[url] = Image.open(io.BytesIO(resp.read())).convert(
                        "RGBA"
                    )
            except Exception:
                return None
        return OffsetImage(logo_cache[url], zoom=0.045)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=130)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    cpoe_mid = qualified_qbs["avg_cpoe"].median()
    epa_mid = qualified_qbs["epa_per_pass"].median()

    ax.axvline(
        cpoe_mid, color="#6c757d", linestyle="--", linewidth=1.1, alpha=0.7
    )
    ax.axhline(
        epa_mid, color="#6c757d", linestyle="--", linewidth=1.1, alpha=0.7
    )

    texts = []
    for _, row in qualified_qbs.iterrows():
        x, y = row["avg_cpoe"], row["epa_per_pass"]
        logo_obj = fetch_logo(row["team_logo_espn"])
        if logo_obj:
            ax.add_artist(
                AnnotationBbox(logo_obj, (x, y), frameon=False, pad=0, zorder=3)
            )
        else:
            ax.scatter(x, y, color="#495057", s=90, zorder=3)

        texts.append(
            ax.text(
                x,
                y + 0.015,
                row["passer_player_name"],
                fontsize=8,
                fontweight="bold",
                ha="center",
            )
        )

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="#ced4da", lw=0.5))

    ax.set_xlabel("Completion Percentage Over Expectation (CPOE %)")
    ax.set_ylabel("EPA per Pass Attempt")
    ax.grid(True, linestyle=":", alpha=0.4, color="#ced4da")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    st.pyplot(fig)

    with st.expander("View Underlying QB Data"):
        st.dataframe(
            qualified_qbs[
                ["passer_player_name", "posteam", "attempts", "avg_cpoe", "epa_per_pass"]
            ].sort_values(by="epa_per_pass", ascending=False)
        )

# -------------------------------------------------------------
# TAB 2: Early vs. Late Downs
# -------------------------------------------------------------
with tab2:
    st.subheader("Team Down-to-Down Efficiency")
    clean_pbp["down_group"] = np.where(
        clean_pbp["down"].isin([1, 2]), "early_down", "late_down"
    )

    team_splits = (
        clean_pbp.groupby(["posteam", "down_group"])
        .agg(
            plays=("epa", "count"),
            epa_per_play=("epa", "mean"),
            success_rate=("success", "mean"),
        )
        .reset_index()
    )

    pivot = team_splits.pivot(
        index="posteam",
        columns="down_group",
        values=["epa_per_play", "success_rate"],
    )
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"posteam": "team"})

    pivot["early_down_epa_per_play"] = pivot["early_down_epa_per_play"].round(3)
    pivot["late_down_epa_per_play"] = pivot["late_down_epa_per_play"].round(3)
    pivot["early_down_success_rate"] = (
        pivot["early_down_success_rate"] * 100
    ).round(1)
    pivot["late_down_success_rate"] = (
        pivot["late_down_success_rate"] * 100
    ).round(1)
    pivot["late_early_epa_gap"] = (
        pivot["late_down_epa_per_play"] - pivot["early_down_epa_per_play"]
    ).round(3)

    st.dataframe(
        pivot.sort_values(by="early_down_epa_per_play", ascending=False),
        use_container_width=True,
    )

# -------------------------------------------------------------
# TAB 3: 3rd Down Scrambles
# -------------------------------------------------------------
with tab3:
    st.subheader("3rd-Down Scramble Conversion Efficiency")
    third_downs = clean_pbp[
        (clean_pbp["down"] == 3) & (clean_pbp["qb_dropback"] == 1)
    ]

    scramble_stats = (
        third_downs.groupby(["passer_player_name", "posteam"])
        .agg(
            dropbacks=("qb_dropback", "count"),
            scrambles=("qb_scramble", "sum"),
            scramble_conv_rate=(
                "third_down_converted",
                lambda x: x[third_downs.loc[x.index, "qb_scramble"] == 1].mean(),
            ),
            scramble_epa=(
                "epa",
                lambda x: x[third_downs.loc[x.index, "qb_scramble"] == 1].mean(),
            ),
        )
        .reset_index()
    )

    scramble_stats["scramble_rate"] = (
        scramble_stats["scrambles"] / scramble_stats["dropbacks"] * 100
    ).round(1)
    scramble_stats["scramble_conv_rate"] = (
        scramble_stats["scramble_conv_rate"] * 100
    ).round(1)
    scramble_stats["scramble_epa"] = scramble_stats["scramble_epa"].round(3)

    st.dataframe(
        scramble_stats[scramble_stats["dropbacks"] >= 25].sort_values(
            by="scramble_rate", ascending=False
        ),
        use_container_width=True,
    )