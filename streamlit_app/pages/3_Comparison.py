import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

st.title("📊 Model Comparison")
st.write("Compare performance metrics across all classification models.")
st.write("---")

metrics_data = {
    "CNN": {
        "Accuracy": 0.7231,
        "F1": 0.7145,
        "Precision": 0.7118,
        "Recall": 0.7231
    },
    "ResNet-50": {
        "Accuracy": 0.9457,
        "F1": 0.9450,
        "Precision": 0.9462,
        "Recall": 0.9457
    },
    "MobileNet-V2": {
        "Accuracy": 0.8760,
        "F1": 0.8754,
        "Precision": 0.8771,
        "Recall": 0.8760
    },
    "EfficientNet-B0": {
        "Accuracy": 0.8915,
        "F1": 0.8908,
        "Precision": 0.8926,
        "Recall": 0.8915
    },
    "ViT": {
        "Accuracy": 0.9225,
        "F1": 0.9218,
        "Precision": 0.9234,
        "Recall": 0.9225
    },
    "Swin": {
        "Accuracy": 0.9380,
        "F1": 0.9372,
        "Precision": 0.9385,
        "Recall": 0.9380
    },
    "DeiT": {
        "Accuracy": 0.9302,
        "F1": 0.9296,
        "Precision": 0.9311,
        "Recall": 0.9302
    },
}

df = pd.DataFrame(metrics_data).T.reset_index().rename(columns={"index": "Model"})

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Performance Metrics Table")

    styled_df = df.style.format({
        "Accuracy": "{:.4f}",
        "F1": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}"
    }).highlight_max(
        subset=["Accuracy", "F1", "Precision", "Recall"],
        color="#90EE90",
        axis=0
    ).highlight_min(
        subset=["Accuracy", "F1", "Precision", "Recall"],
        color="#FFB6C6",
        axis=0
    )

    st.dataframe(styled_df, use_container_width=True)

with col2:
    st.subheader("📈 Key Stats")

    best_model_acc = df.loc[df["Accuracy"].idxmax()]
    best_model_f1 = df.loc[df["F1"].idxmax()]

    st.metric("Best Accuracy", best_model_acc["Model"], f"({best_model_acc['Accuracy']:.4f})")
    st.metric("Best F1 Score", best_model_f1["Model"], f"({best_model_f1['F1']:.4f})")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Accuracy Comparison")
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    models = df["Model"].tolist()
    accuracy_vals = df["Accuracy"].tolist()
    colors = ["#2ecc71" if acc == max(accuracy_vals) else "#3498db" for acc in accuracy_vals]

    bars1 = ax1.barh(models, accuracy_vals, color=colors, edgecolor="black", linewidth=1.2)

    for bar, val in zip(bars1, accuracy_vals):
        ax1.text(val - 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="right", va="center", fontweight="bold", color="white", fontsize=9)

    ax1.set_xlim(0.7, 1.0)
    ax1.set_xlabel("Accuracy", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3, linestyle="--")
    ax1.set_title("Accuracy Scores", fontweight="bold", fontsize=12)

    fig1.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig1, use_container_width=True)

with col2:
    st.subheader("F1 Score Comparison")
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    f1_vals = df["F1"].tolist()
    colors = ["#e74c3c" if f1 == min(f1_vals) else "#9b59b6" for f1 in f1_vals]

    bars2 = ax2.barh(models, f1_vals, color=colors, edgecolor="black", linewidth=1.2)

    for bar, val in zip(bars2, f1_vals):
        ax2.text(val - 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", ha="right", va="center", fontweight="bold", color="white", fontsize=9)

    ax2.set_xlim(0.7, 1.0)
    ax2.set_xlabel("F1 Score", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    ax2.set_title("F1 Scores", fontweight="bold", fontsize=12)

    fig2.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Precision vs Recall")
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars_prec = ax3.bar(x - width / 2, df["Precision"], width, label="Precision", color="#3498db", edgecolor="black")
    bars_rec = ax3.bar(x + width / 2, df["Recall"], width, label="Recall", color="#e74c3c", edgecolor="black")

    ax3.set_ylabel("Score", fontweight="bold")
    ax3.set_title("Precision vs Recall", fontweight="bold", fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha="right")
    ax3.legend()
    ax3.set_ylim(0.7, 1.0)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    fig3.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)

with col2:
    st.subheader("Model Categories")

    fig4, ax4 = plt.subplots(figsize=(8, 6))

    phases = {
        "Phase 1: CNN": df[df["Model"] == "CNN"]["Accuracy"].values[0],
        "Phase 2: Transfer\nLearning": df[df["Model"].isin(["ResNet-50", "MobileNet-V2", "EfficientNet-B0"])]["Accuracy"].mean(),
        "Phase 3: Vision\nTransformers": df[df["Model"].isin(["ViT", "Swin", "DeiT"])]["Accuracy"].mean(),
    }

    colors_phase = ["#4C72B0", "#55A868", "#C44E52"]
    bars4 = ax4.bar(phases.keys(), phases.values(), color=colors_phase, edgecolor="black", linewidth=1.5)

    for bar, val in zip(bars4, phases.values()):
        ax4.text(bar.get_x() + bar.get_width() / 2, val - 0.02,
                f"{val:.4f}", ha="center", va="top", fontweight="bold", color="white", fontsize=10)

    ax4.set_ylabel("Mean Accuracy", fontweight="bold")
    ax4.set_title("Performance by Development Phase", fontweight="bold", fontsize=12)
    ax4.set_ylim(0.7, 1.0)
    ax4.grid(axis="y", alpha=0.3, linestyle="--")

    fig4.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig4, use_container_width=True)

st.divider()

with st.expander("💡 Insights & Recommendations"):
    st.write("""
    ### 🏆 Top Performer
    **ResNet-50** achieves the highest accuracy (94.57%) and is recommended for production deployment.

    ### 📊 Model Performance Tiers
    - **Tier 1 (94%+)**: ResNet-50, DeiT, Swin
    - **Tier 2 (92%)**: ViT
    - **Tier 3 (88%+)**: EfficientNet-B0, MobileNet-V2
    - **Tier 4 (72%)**: Custom CNN (baseline)

    ### 💡 Key Observations
    - **Vision Transformers** (ViT, Swin, DeiT) generally outperform traditional CNNs
    - **ResNet-50** provides the best accuracy-to-speed tradeoff
    - **MobileNet-V2** is ideal for edge deployment (88.76% accuracy, smaller model)
    - All transfer learning models significantly outperform the baseline CNN

    ### 🎯 Recommendations
    - **Production**: Use ResNet-50 for best accuracy
    - **Edge Devices**: Use MobileNet-V2 for mobile/embedded systems
    - **Research**: Explore Vision Transformers for state-of-the-art results
    """)
