import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io, os

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Precision Therapeutics – Polymer AI Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# THEME TOGGLE
# ==================================================
dark_mode = st.sidebar.toggle("Dark Mode")

if dark_mode:
    bg, text, card, border = "#0e1117", "#f5f5f5", "#1c1f26", "#2b2f3a"
else:
    bg, text, card, border = "#f4f7fb", "#111111", "#ffffff", "#dcdcdc"

st.markdown(f"""
<style>
.stApp {{ background-color:{bg}; color:{text}; }}
h1,h2,h3,h4,h5,h6,p,li,span,label {{ color:{text}!important; }}
section[data-testid="stSidebar"] {{ background:{card}; }}
.card {{ background:{card}; border:1px solid {border};
        padding:20px; border-radius:12px; margin-bottom:18px; }}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA (SAFE + HUMAN ONLY)
# ==================================================
@st.cache_data
def load_data():
    base = os.path.join(os.getcwd(), "data")

    genes = pd.read_excel(os.path.join(base, "patient_gene_data.xlsx"))
    drugs = pd.read_excel(os.path.join(base, "human_useful_drug_data.xlsx"))
    polymers = pd.read_excel(os.path.join(base, "Overall_Polymer Data.xlsx"))

    # Normalize gene headers
    genes.columns = [c.upper() if c != "Patient_ID" else c for c in genes.columns]

    # Normalize drug genes
    drugs["GENE"] = (
        drugs["GENE"].astype(str)
        .str.upper()
        .str.replace("_HUMAN", "", regex=False)
        .str.replace("P53", "TP53", regex=False)
        .str.strip()
    )

    drugs["ACT_VALUE"] = pd.to_numeric(drugs["ACT_VALUE"], errors="coerce")
    drugs = drugs.dropna(subset=["ACT_VALUE"])
    drugs["pIC50"] = -np.log10(drugs["ACT_VALUE"].clip(lower=1e-12))

    # Polymer normalization
    polymers["Release_Norm"] = polymers["Drug Release %"] / polymers["Drug Release %"].max()
    polymers["Bio_Norm"] = polymers["Bio Score"] / polymers["Bio Score"].max()

    return genes, drugs, polymers


genes, drugs, polymers = load_data()
GENE_COLUMNS = [c for c in genes.columns if c != "Patient_ID"]

# ==================================================
# AI SCORE (ROBUST)
# ==================================================
def ai_score(gene_vals, drug, polymer):
    gene_signal = np.mean(gene_vals)
    return (
        0.5 * gene_signal +
        0.3 * drug["pIC50"] +
        0.2 * (0.6 * polymer["Release_Norm"] + 0.4 * polymer["Bio_Norm"])
    )

# ==================================================
# NAVIGATION
# ==================================================
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Patient Gene Dataset",
        "Drug Dataset",
        "Polymer Dataset",
        "Gene Analysis",
        "Patient Gene Comparison",
        "AI Therapy Prediction",
        "Scenario Simulation"
    ]
)

# ==================================================
# HOME
# ==================================================
if page == "Home":
    st.title("Precision Therapeutics – Polymer AI Platform")
    st.caption("AI-assisted integration of patient genomics, drug potency, and polymer optimization")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    **Platform Capabilities**
    - Patient-specific gene expression analysis  
    - Human drug–gene intelligence  
    - AI-driven drug–polymer ranking  
    - Scenario simulation and explainable scoring  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# DATASET VIEWS
# ==================================================
elif page == "Patient Gene Dataset":
    st.title("Patient Gene Dataset")
    st.caption("Individual-level gene expression profiles")
    st.dataframe(genes, use_container_width=True)

elif page == "Drug Dataset":
    st.title("Drug Dataset (Human)")
    st.caption("Human-targeted drug–gene–potency information")
    st.dataframe(drugs, use_container_width=True)

elif page == "Polymer Dataset":
    st.title("Polymer Dataset")
    st.caption("Polymer release and bioactivity properties")
    st.dataframe(polymers, use_container_width=True)

# ==================================================
# GENE ANALYSIS
# ==================================================
elif page == "Gene Analysis":
    st.title("Patient Gene Expression Analysis")
    st.caption("Bar plot and heatmap of selected patient genes")

    pid = st.selectbox("Select Patient", genes["Patient_ID"])
    row = genes.loc[genes["Patient_ID"] == pid, GENE_COLUMNS]

    st.plotly_chart(px.bar(row.T, title="Gene Expression"), use_container_width=True)
    st.plotly_chart(px.imshow(row.values, aspect="auto"), use_container_width=True)

# ==================================================
# PATIENT COMPARISON (BAR + HEATMAP)
# ==================================================
elif page == "Patient Gene Comparison":
    st.title("Multi-Patient Gene Comparison")
    st.caption("Grouped bar graph and heatmap comparison")

    selected = st.multiselect("Select Patients (2 or more)", genes["Patient_ID"])

    if len(selected) < 2:
        st.info("Please select at least two patients.")
    else:
        comp = genes[genes["Patient_ID"].isin(selected)]

        long_df = comp.melt(
            id_vars="Patient_ID",
            value_vars=GENE_COLUMNS,
            var_name="Gene",
            value_name="Expression"
        )

        # Bar graph
        fig_bar = px.bar(
            long_df,
            x="Gene",
            y="Expression",
            color="Patient_ID",
            barmode="group",
            title="Gene-wise Expression Comparison",
            labels={"Patient_ID": "Patient"}
        )
        fig_bar.update_layout(legend_title_text="Patient ID", xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Heatmap
        heat = comp.set_index("Patient_ID")[GENE_COLUMNS]
        fig_heat = px.imshow(
            heat.values,
            x=GENE_COLUMNS,
            y=heat.index,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels=dict(x="Gene", y="Patient", color="Expression")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ==================================================
# AI THERAPY PREDICTION
# ==================================================
elif page == "AI Therapy Prediction":
    st.title("AI Therapy Prediction")
    st.caption("Patient-specific drug–polymer optimization")

    pid = st.selectbox("Select Patient", genes["Patient_ID"])
    genes_sel = st.multiselect("Select Genes", GENE_COLUMNS, GENE_COLUMNS[:5])
    polymers_sel = st.multiselect("Select Polymers", polymers["Polymer"])

    if st.button("Run Prediction"):
        gvals = genes.loc[genes["Patient_ID"] == pid, genes_sel].values.flatten()

        matched = drugs[drugs["GENE"].isin(genes_sel)]
        if matched.empty:
            matched = drugs.sort_values("pIC50", ascending=False).head(20)

        rows = []
        for _, d in matched.iterrows():
            for _, p in polymers[polymers["Polymer"].isin(polymers_sel)].iterrows():
                rows.append([d["DRUG_NAME"], p["Polymer"], ai_score(gvals, d, p)])

        df = pd.DataFrame(rows, columns=["Drug", "Polymer", "Score"])
        st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True)

# ==================================================
# SCENARIO SIMULATION
# ==================================================
elif page == "Scenario Simulation":
    st.title("Scenario Simulation")
    st.caption("What-if gene perturbation analysis")

    genes_sel = st.multiselect("Genes", GENE_COLUMNS, GENE_COLUMNS[:5])
    gvals = [st.slider(g, 0.0, 1.0, 0.5) for g in genes_sel]
    polymers_sel = st.multiselect("Polymers", polymers["Polymer"])

    if st.button("Simulate"):
        matched = drugs[drugs["GENE"].isin(genes_sel)]
        if matched.empty:
            matched = drugs.sort_values("pIC50", ascending=False).head(20)

        rows = []
        for _, d in matched.iterrows():
            for _, p in polymers[polymers["Polymer"].isin(polymers_sel)].iterrows():
                rows.append([d["DRUG_NAME"], p["Polymer"], ai_score(gvals, d, p)])

        st.dataframe(pd.DataFrame(rows, columns=["Drug", "Polymer", "Score"]),
                     use_container_width=True)

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.warning("Research-grade AI decision-support system. Not for clinical use.")
