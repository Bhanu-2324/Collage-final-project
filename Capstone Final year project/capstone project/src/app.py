import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier


sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_and_preprocess

st.set_page_config(
    page_title="Youth Mental Health & AI Analysis",
    page_icon="🧠",
    layout="wide"
)
@st.cache_data(show_spinner="Connecting to Neural Database...")
def get_cached_data():
    """Loads and preprocesses the data."""
    return load_and_preprocess()

df = get_cached_data()


action_col1, action_col2 = st.columns([3, 1])

with action_col1:
    st.markdown("""
    <div style="
        background: rgba(15, 23, 42, 0.85);
        padding: 18px 22px;
        border-radius: 16px;
        border: 1px solid rgba(96, 165, 250, 0.35);
        box-shadow: 0 0 18px rgba(59, 130, 246, 0.12);
        margin-bottom: 20px;
    ">
        <div style="font-size: 18px; font-weight: 600; color: #60a5fa;">
            🚀 Want to contribute?
        </div>
        <div style="font-size: 14px; color: #cbd5e1; margin-top: 6px;">
            Try giving the Google Form a hit with your response, then refresh to view the latest live dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

with action_col2:
    st.link_button(
        "📝 Open Google Form",
        "https://forms.gle/NFndQVX4ZpXZqcJ26",
        use_container_width=True
    )

refresh_col1, refresh_col2, refresh_col3 = st.columns([1, 1, 2])

with refresh_col1:
    if st.button("🔄 Refresh Dataset", use_container_width=True):
        get_cached_data.clear()
        st.rerun()



st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Global Text Colors */
    h1, h2, h3, h4, p, span, label {
        color: #f8fafc !important;
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Filter Container Styling */
    .filter-container {
        background: rgba(15, 23, 42, 0.8);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #3b82f6;
        margin-bottom: 25px;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
    }
    
    /* Insight Card Styling */
    .insight-card {
        background: rgba(59, 130, 246, 0.1);
        padding: 18px;
        border-radius: 12px;
        border-left: 5px solid #60a5fa;
        margin-top: 10px;
    }
    
    /* Slider Customization - FIXED VISIBILITY */
    /* The track (background) */
    .stSlider [data-baseweb="slider"] > div:first-child {
        background: rgba(255, 255, 255, 0.1);
        height: 8px;
    }
    /* The filled range (the "bar") */
    .stSlider [data-baseweb="slider"] > div > div {
        background: #f87171 !important; /* Vibrant Coral/Red for the selected range */
    }
    /* The handles (the dots) */
    .stSlider [data-baseweb="thumb"] {
        background-color: #f8fafc !important;
        border: 2px solid #f87171 !important;
        height: 20px;
        width: 20px;
    }
    
    /* Divider Color */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Expander Styling */
    div[data-testid="stExpander"] {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center; color: #60a5fa !important;'>🧠 Youth Mental Health: Digital Impact</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.8;'>Analyzing the intersection of Screen Time, AI Reliance, and Stress Levels</p>", unsafe_allow_html=True)

if not df.empty:
    # Top Filter Container
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        st.subheader("🎯 Demographic Control Panel")
        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())
        age_range = st.slider("Target Age Bracket", min_age, max_age, (min_age, max_age))
        
        st.divider()
        
        # Filter Statistics
        stat_col1, stat_col2, stat_col3 = st.columns([1, 1, 2])
        filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
        percent_of_total = (len(filtered_df) / len(df)) * 100
        
        with stat_col1:
            st.markdown(f"**Segment Size:** <span style='color:#60a5fa'>{len(filtered_df)}</span> users", unsafe_allow_html=True)
        with stat_col2:
            st.markdown(f"**Coverage:** <span style='color:#60a5fa'>{percent_of_total:.1f}%</span>", unsafe_allow_html=True)
        with stat_col3:
            st.progress(percent_of_total / 100)
            
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error("Neural sync failed. Please check data source connection.")
    st.stop()

st.markdown(f"#### 📉 Analytics Snapshot: Ages {age_range[0]} - {age_range[1]}")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Database", len(df))
with c2:
    st.metric("Active Segment", len(filtered_df), delta=f"{len(filtered_df) - len(df)}")
with c3:
    avg_screen = filtered_df["DailyScreenTime(hours)"].mean()
    st.metric("Avg Digital Usage", f"{avg_screen:.1f}h/day")
with c4:
    avg_stress = filtered_df["StressLevel"].mean()
    st.metric("Avg Stress Index", f"{avg_stress:.1f}/10")

st.divider()
st.subheader("⚖️ Stress Architecture")


plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = '#1e293b'
plt.rcParams['figure.facecolor'] = '#1e293b'
plt.rcParams['text.color'] = '#f8fafc'


ml_ready = df.select_dtypes(include=['number']).copy()
for col in ml_ready.columns:
    ml_ready[col] = ml_ready[col].fillna(ml_ready[col].median())

if len(ml_ready) > 10: 
    y = ml_ready["StressLevel"]
    X = ml_ready.drop(columns=["StressLevel", "ProductivityScore", "MentalHealthImpact"], errors='ignore')

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    m_col1, m_col2 = st.columns([3, 2])

    with m_col1:
        st.markdown("**Predictive Driver Weights**")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        importances.plot(kind='barh', color='#3b82f6', ax=ax_imp)
        ax_imp.set_title(f"Global Stress Drivers", fontsize=12, color='#60a5fa')
        ax_imp.spines['top'].set_visible(False)
        ax_imp.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_imp)

    with m_col2:
        st.markdown("**Segment Interaction Matrix**")
        corr_cols = ["Age", "DailyScreenTime(hours)", "SleepHours", "StressLevel", "AI_Usage_Time(hours/day)", "ProductivityScore"]
        existing_cols = [c for c in corr_cols if c in filtered_df.columns]
        corr_matrix = filtered_df[existing_cols].select_dtypes(include=['number']).corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(8, 8))
        im = ax_corr.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(corr_matrix.columns)))
        ax_corr.set_yticks(range(len(corr_matrix.columns)))
        ax_corr.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", color='#f8fafc')
        ax_corr.set_yticklabels(corr_matrix.columns, color='#f8fafc')
        
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax_corr.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="white")
        
        plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig_corr)
else:
    st.warning("Insufficient data for neural modeling.")

st.divider()
st.subheader("📉 Productivity/Stress Correlation")
p_col1, p_col2 = st.columns([2, 1])

with p_col1:
    fig_prod, ax_prod = plt.subplots(figsize=(10, 5))
    ax_prod.scatter(filtered_df["StressLevel"], filtered_df["ProductivityScore"], alpha=0.6, color='#60a5fa', edgecolors='white', linewidth=0.5)
    

    z = np.polyfit(filtered_df["StressLevel"], filtered_df["ProductivityScore"], 1)
    p = np.poly1d(z)
    ax_prod.plot(filtered_df["StressLevel"], p(filtered_df["StressLevel"]), color='#f87171', linestyle='--', linewidth=2)
    
    ax_prod.set_xlabel("Stress Level (0-10)", color='#f8fafc')
    ax_prod.set_ylabel("Productivity Score (0-10)", color='#f8fafc')
    ax_prod.set_title("Stress vs. Output Trend", color='#60a5fa')
    st.pyplot(fig_prod)

with p_col2:
    st.markdown("### Mental Sentiment Index")
    if "MentalHealthImpact" in filtered_df.columns:
        impact_counts = filtered_df["MentalHealthImpact"].value_counts(normalize=True) * 100
        for impact, val in impact_counts.items():


            impact_label_map = {
            1: "Positive",
            0: "Neutral",
            2: "Negative"
        }       

        impact_color_map = {
            "Positive": "#4ade80",
            "Neutral": "#94a3b8",
            "Negative": "#f87171"
        }

        for impact, val in impact_counts.items():
            label = impact_label_map.get(impact, "Unknown")
            color = impact_color_map.get(label, "#94a3b8")

            st.markdown(f"**{label}:** {val:.1f}%")
            st.progress(val / 100)


          
    
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.write("💡 **System Insight:**")
    if avg_stress > 7:
        st.write("CRITICAL: High stress detected in this demographic. Intervention recommended.")
    elif avg_screen > 6:
        st.write("WARNING: Excessive digital exposure. Monitor screen-to-sleep ratios.")
    else:
        st.write("STABLE: Habits are within nominal range for this segment.")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.subheader("📊 Habit Distributions")
d_col1, d_col2 = st.columns(2)

with d_col1:
    fig_hist, ax_hist = plt.subplots()
    filtered_df["DailyScreenTime(hours)"].hist(bins=15, color='#3b82f6', ax=ax_hist, grid=False, rwidth=0.9)
    ax_hist.set_title(f"Screen Time Spread", color='#60a5fa')
    ax_hist.set_xlabel("Hours per Day")
    st.pyplot(fig_hist)

with d_col2:
    fig_box, ax_box = plt.subplots()
    if "StressLevel" in filtered_df.columns:
        # Customizing boxplot colors
        bp = filtered_df.boxplot(column='SleepHours', by='StressLevel', ax=ax_box, patch_artist=True)
        plt.suptitle("") 
        ax_box.set_title(f"Sleep Variance by Stress", color='#60a5fa')
        ax_box.set_ylabel("Hours of Sleep")
        st.pyplot(fig_box)

st.divider()
st.subheader("🤖 AI Reliance Analytics")

denom = (filtered_df["Problems_Solved_With_AI"] + filtered_df["Problems_Solved_Before_AI"])
filtered_df["AI_Reliance"] = filtered_df["Problems_Solved_With_AI"] / (denom.replace(0, 1))

b1, b2, b3 = st.columns(3)
with b1:
    high_ai = len(filtered_df[filtered_df["AI_Reliance"] > 0.7])
    st.metric("Neural Reliant", high_ai, "Critical Segment")
with b2:
    balanced_ai = len(filtered_df[(filtered_df["AI_Reliance"] <= 0.7) & (filtered_df["AI_Reliance"] >= 0.3)])
    st.metric("Hybrid Users", balanced_ai, "Optimal Balance")
with b3:
    self_reliant = len(filtered_df[filtered_df["AI_Reliance"] < 0.3])
    st.metric("Analog Solvers", self_reliant, "Traditional Segment")

st.divider()
with st.expander("🔍 Neural Data Explorer & Export"):
    st.write(f"Accessing raw encrypted segments for ages {age_range[0]} to {age_range[1]}")
    st.dataframe(
    filtered_df.drop(columns=['AI_Reliance'], errors='ignore').head(20),
    width="stretch"
)
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Export Encrypted Segment (.CSV)", data=csv, file_name=f'neural_data_ages_{age_range[0]}_{age_range[1]}.csv', mime='text/csv')

st.markdown("---")
st.caption(f"Youth Mental Health AI | Processing {len(filtered_df)} active demographic nodes.")