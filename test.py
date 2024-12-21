import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joypy
from scipy.stats import iqr
import plotly.express as px

st.title("Data Processing, Analysis and Visualization for Vietnam High School Graduation Examination")

# Load data
@st.cache_data
def load_data():
    df_score_1 = pd.read_csv('vnhsge-2018_score_1.csv')
    df_score_2 = pd.read_csv('vnhsge-2018_score_2.csv')
    df_province_region = pd.read_csv('vnhsge_province_region.csv')

    # Merge score dataframes
    df_score = pd.concat([df_score_1, df_score_2]).drop_duplicates()
    df_score = df_score[df_score['score'] > 0]  # Filter valid scores

    # Add province code
    def extract_prefix(candidate_id):
        if len(str(candidate_id)) == 7:
            return int(str(candidate_id)[:1])
        elif len(str(candidate_id)) == 8:
            return int(str(candidate_id)[:2])
        return None

    df_score['province_code'] = df_score['candidate_id'].apply(extract_prefix)

    # Merge with region data
    merged_df = pd.merge(df_score, df_province_region, on='province_code', how='inner')
    return merged_df


merged_df = load_data()

# Navigation bar
page = st.sidebar.radio("Navigation", ["Polar Chart", "Joyplot", "Boxplots", "JointGrid"])

# 1. Polar Chart Page
if page == "Polar Chart":
    st.header("Polar Chart: Subject Selection Rates by Region")
    st.write("""
    This chart shows the percentage of candidates selecting specific subjects for admission by region. 
    It provides insights into regional preferences for different subjects.
    """)
    
    # Count total candidates by region and subject_id
    total_candidates = merged_df.groupby(['region', 'subject_id'])['candidate_id'].count().reset_index()
    total_candidates.rename(columns={'candidate_id': 'total_candidates'}, inplace=True)
    total_candidates = total_candidates[total_candidates['subject_id'] != 'GDCD']

    # Count subjects by region where is_chosen equals 1
    subject_count_by_region_chosen = merged_df[merged_df['is_chosen'] == 1].groupby(['er_code', 'region', 'subject_id'])['candidate_id'].count().reset_index()
    subject_count_by_region_chosen.rename(columns={'candidate_id': 'total_candidates_chosen'}, inplace=True)

    # Merge and calculate percentages
    df = pd.merge(total_candidates, subject_count_by_region_chosen, on=['region', 'subject_id'])
    df['percentage'] = round((df['total_candidates_chosen'] / df['total_candidates']) * 100, 2)

    # Plot Polar Chart
    fig = px.line_polar(df, r='percentage', theta='subject_id', line_close=True, color='region',
                        title="Subject Selection Rates by Region")
    st.plotly_chart(fig)
    st.write(df)

# 2. Joyplot Page
elif page == "Joyplot":
    st.header("Joyplot: Scores Distribution by Region")
    st.write("""
    The Joyplot visualizes the distribution of scores for different regions, categorized into subjects chosen 
    for admission and non-admission purposes.
    """)
    subjects = merged_df['subject_id'].unique().tolist()
    selected_subject = st.selectbox("Select Subject", subjects)

    # Joyplot
    if selected_subject:
        col1, col2 = st.columns(2)
        for index, col in enumerate([col1, col2]):
            with col:
                filtered_df = merged_df[(merged_df['subject_id'] == selected_subject) & (merged_df['is_chosen'] == index)]
                plt.figure()
                fig, ax = joypy.joyplot(filtered_df, by="region", column="score", linecolor="white")
                plt.title(f"{'Admission' if index else 'Non-admission'} Joyplot")
                st.pyplot(plt)

# 3. Boxplots Page
elif page == "Boxplots":
    st.header("Boxplots: Scores Distribution by Subject and Block")
    st.write("""
    Boxplots show the distribution of scores across different subjects and blocks. 
    They help identify trends, outliers, and variations in performance.
    """)

    def calculate_stats(df, type):
        stats = []
        id = f'{type}_id'.lower()
        for x in df[id].unique():
            filtered = df[df[id] == x]
            stats.append({
                type: x,
                "Num Rows": len(filtered),
                "Mean": round(filtered['score'].mean(), 2),
                "Std Dev": round(filtered['score'].std(), 2),
                "Percent Below 5": round((filtered['score'] < 5).mean() * 100, 2)
            })
        return pd.DataFrame(stats)

    def draw_boxplot(df, type):
        id = f'{type}_id'.lower()
        plt.figure()
        sns.boxplot(data=df, x=id, y='score', hue=id, palette="Set3")
        plt.title(f"{type} Boxplot")
        plt.xlabel(type)
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        return plt

    df_block_score = pd.read_csv('vnhsge-2018_block_score.csv')

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(draw_boxplot(df_block_score, 'Block'))
        st.dataframe(calculate_stats(df_block_score, 'Block'))
    with col2:
        st.pyplot(draw_boxplot(merged_df, 'Subject'))
        st.dataframe(calculate_stats(merged_df, 'Subject'))

# 4. JointGrid Page
elif page == "JointGrid":
    st.header("JointGrid: Relationship between Score Range and Average Score")
    st.write("""
    This JointGrid chart explores the relationship between score ranges and average scores. 
    It combines scatter plots and histograms to show both correlation and distribution.
    """)

    df_grad = pd.read_csv('vnhsge-2018_avg.csv')

    # Round function for score ranges and average scores
    def round_to_nearest_half(x):
        return np.round(x * 2) / 2

    df_grad['score_range'] = df_grad['score_range'].apply(round_to_nearest_half)
    df_grad['avg_score'] = df_grad['avg_score'].apply(round_to_nearest_half)

    plt.figure(figsize=(8, 6))
    g = sns.jointplot(data=df_grad, x="score_range", y="avg_score", kind="hist", binwidth=0.5)
    g.plot_marginals(sns.histplot, binwidth=0.5)
    g.refline(x=2, y=5, color='red')
    st.pyplot(plt)
    st.write(df_grad)
