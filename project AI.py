import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(
    page_title="SoluChain - AI Organic Compound Solubility Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. CUSTOM CSS: TEMA BIRU - PUTIH (ELEGANT LIGHT THEME)
# ==========================================
st.markdown(
    """
    <style>
    /* Latar Belakang Utama */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Header Utama SoluChain */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 28px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .main-header p {
        color: #dbeafe;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Kartu Metrik KPI */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 4px solid #2563eb;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    .metric-value { 
        font-size: 20px; 
        font-weight: 800; 
        color: #1e3a8a; 
    }
    .metric-label { 
        font-size: 12px; 
        color: #64748b; 
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Kotak Log Inferensi */
    .rule-box {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #1e40af;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* Hero Banner Landing Page */
    .hero-card {
        background-color: #ffffff;
        border: 2px dashed #93c5fd;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 3. DATABASE SENYAWA LENGKAP (45 SENYAWA)
# =========================================================
AQSOL_IUPAC_DATABASE = {
    # --- 1. ALCOHOL ---
    "butan-1-ol": {
        "smiles": "CCCCO",
        "Group": "Alcohol",
        "pKa1": 16.1,
        "Solubility": -0.050409,
        "MolLogP": 0.88,
    },
    "benzyl alcohol": {
        "smiles": "C1=CC=C(C=C1)CO",
        "Group": "Alcohol",
        "pKa1": 15.4,
        "Solubility": -0.4319,
        "MolLogP": 1.10,
    },
    "1-propanol": {
        "smiles": "CCCO",
        "Group": "Alcohol",
        "pKa1": 16.1,
        "Solubility": 0.62,
        "MolLogP": 0.25,
    },
    "ethylene glycol": {
        "smiles": "OCCO",
        "Group": "Alcohol",
        "pKa1": 15.4,
        "Solubility": 1.2071,
        "MolLogP": -1.36,
    },
    "2-butanol": {
        "smiles": "CCC(C)O",
        "Group": "Alcohol",
        "pKa1": 17.6,
        "Solubility": 0.3877,
        "MolLogP": 0.65,
    },
    # --- 2. CARBOXYLIC ACID ---
    "propanoic acid": {
        "smiles": "CCC(=O)O",
        "Group": "Carboxylic Acid",
        "pKa1": 4.86,
        "Solubility": 1.130305,
        "MolLogP": 0.33,
    },
    "acetic acid": {
        "smiles": "CC(=O)O",
        "Group": "Carboxylic Acid",
        "pKa1": 4.73,
        "Solubility": 1.001718,
        "MolLogP": -0.17,
    },
    "benzoic acid": {
        "smiles": "C1=CC=C(C=C1)C(=O)O",
        "Group": "Carboxylic Acid",
        "pKa1": 4.13,
        "Solubility": -1.61993,
        "MolLogP": 1.87,
    },
    "cholic acid": {
        "smiles": "CC(CCC(=O)O)C1CCC2C1(CCC3C2C(CC4C3(CCC(C4)O)C)O)O",
        "Group": "Carboxylic Acid",
        "pKa1": 4.98,
        "Solubility": -3.3682,
        "MolLogP": 2.02,
    },
    "cumic acid": {
        "smiles": "CC(C)C1=CC=C(C=C1)C(=O)O",
        "Group": "Carboxylic Acid",
        "pKa1": 4.34,
        "Solubility": -3.0364,
        "MolLogP": 2.91,
    },
    # --- 3. ETHER ---
    "allyl ether": {
        "smiles": "C=CCOCC=C",
        "Group": "Ether",
        "pKa1": None,
        "Solubility": -0.0206,
        "MolLogP": 1.36,
    },
    "butoxybenzene": {
        "smiles": "CCCCOC1=CC=CC=C1",
        "Group": "Ether",
        "pKa1": None,
        "Solubility": -3.61325,
        "MolLogP": 3.07,
    },
    "methoxychlor": {
        "smiles": "COC1=CC=C(C=C1)C(C2=CC=C(C=C2)OC)C(Cl)(Cl)Cl",
        "Group": "Ether",
        "pKa1": None,
        "Solubility": -6.5386,
        "MolLogP": 5.78,
    },
    "diethyl ether": {
        "smiles": "CCOCC",
        "Group": "Ether",
        "pKa1": None,
        "Solubility": -0.0889,
        "MolLogP": 0.89,
    },
    "anisole": {
        "smiles": "COC1=CC=CC=C1",
        "Group": "Ether",
        "pKa1": None,
        "Solubility": -1.85,
        "MolLogP": 2.11,
    },
    # --- 4. ESTER ---
    "dimethoate": {
        "smiles": "CNC(=O)CSP(=S)(OC)OC",
        "Group": "Ester",
        "pKa1": None,
        "Solubility": -0.9624,
        "MolLogP": -0.08,
    },
    "cycloate": {
        "smiles": "CCN(C1CCCCC1)C(=O)SCC",
        "Group": "Ester",
        "pKa1": None,
        "Solubility": -3.4037,
        "MolLogP": 3.11,
    },
    "fenthoate": {
        "smiles": "CCOC(=O)C(C1=CC=CC=C1)SP(=S)(OC)OC",
        "Group": "Ester",
        "pKa1": None,
        "Solubility": -4.4643,
        "MolLogP": 3.68,
    },
    "ethyl acetate": {
        "smiles": "CCOC(C)=O",
        "Group": "Ester",
        "pKa1": None,
        "Solubility": -0.025404,
        "MolLogP": 0.73,
    },
    "methyl benzoate": {
        "smiles": "COC(=O)C1=CC=CC=C1",
        "Group": "Ester",
        "pKa1": None,
        "Solubility": -1.811798,
        "MolLogP": 2.12,
    },
    # --- 5. ALDEHYDE ---
    "formaldehyde": {
        "smiles": "C=O",
        "Group": "Aldehyde",
        "pKa1": 12.2,
        "Solubility": 1.1206,
        "MolLogP": 0.35,
    },
    "acetaldehyde": {
        "smiles": "CC=O",
        "Group": "Aldehyde",
        "pKa1": 13.57,
        "Solubility": 1.3561,
        "MolLogP": -0.17,
    },
    "benzaldehyde": {
        "smiles": "C1=CC=C(C=C1)C=O",
        "Group": "Aldehyde",
        "pKa1": 14.9,
        "Solubility": -1.209572,
        "MolLogP": 1.48,
    },
    "4-methylbenzaldehyde": {
        "smiles": "CC1=CC=C(C=C1)C=O",
        "Group": "Aldehyde",
        "pKa1": 15.39,
        "Solubility": -1.723702,
        "MolLogP": 2.01,
    },
    "2,2,2-trichloroacetaldehyde": {
        "smiles": "C(=O)C(Cl)(Cl)Cl",
        "Group": "Aldehyde",
        "pKa1": 9.95,
        "Solubility": -0.691341,
        "MolLogP": 1.42,
    },
    # --- 6. KETONE ---
    "cyclohexanone": {
        "smiles": "O=C1CCCCC1",
        "Group": "Ketone",
        "pKa1": 16.7,
        "Solubility": -0.05737,
        "MolLogP": 0.81,
    },
    "4-methylpent-3-en-2-one": {
        "smiles": "CC(=CC(=O)C)C",
        "Group": "Ketone",
        "pKa1": 20.5,
        "Solubility": -0.56,
        "MolLogP": 1.33,
    },
    "1-methylpyrrolidin-2-one": {
        "smiles": "CN1CCCC1=O",
        "Group": "Ketone",
        "pKa1": 24.0,
        "Solubility": 1.00,
        "MolLogP": -0.38,
    },
    "acetone": {
        "smiles": "CC(=O)C",
        "Group": "Ketone",
        "pKa1": 20.0,
        "Solubility": 1.236,
        "MolLogP": -0.24,
    },
    "1-phenylethan-1-one": {
        "smiles": "CC(=O)C1=CC=CC=C1",
        "Group": "Ketone",
        "pKa1": 19.2,
        "Solubility": -1.280387,
        "MolLogP": 1.58,
    },
    # --- 7. AMINE ---
    "ethylamine": {
        "smiles": "CCN",
        "Group": "Amine",
        "pKa1": 10.79,
        "Solubility": 1.346,
        "MolLogP": -0.13,
    },
    "hexylamine": {
        "smiles": "CCCCCCN",
        "Group": "Amine",
        "pKa1": 10.56,
        "Solubility": -1.1,
        "MolLogP": 2.06,
    },
    "aniline": {
        "smiles": "C1=CC=C(C=C1)N",
        "Group": "Amine",
        "pKa1": 4.60,
        "Solubility": -0.425017,
        "MolLogP": 0.90,
    },
    "pyridine": {
        "smiles": "C1=CC=NC=C1",
        "Group": "Amine",
        "pKa1": 5.25,
        "Solubility": 0.76,
        "MolLogP": 0.65,
    },
    "benzylamine": {
        "smiles": "C1=CC=C(C=C1)CN",
        "Group": "Amine",
        "pKa1": 9.33,
        "Solubility": -1.53,
        "MolLogP": 1.09,
    },
    # --- 8. PHENOL ---
    "phenol": {
        "smiles": "C1=CC=C(C=C1)O",
        "Group": "Phenol",
        "pKa1": 9.98,
        "Solubility": -0.04,
        "MolLogP": 1.46,
    },
    "4-methylphenol": {
        "smiles": "CC1=CC=C(C=C1)O",
        "Group": "Phenol",
        "pKa1": 10.26,
        "Solubility": -0.701548,
        "MolLogP": 1.94,
    },
    "o-aminophenol": {
        "smiles": "C1=CC=C(C(=C1)N)O",
        "Group": "Phenol",
        "pKa1": 9.71,
        "Solubility": -0.7369,
        "MolLogP": 0.62,
    },
    "4-nitrophenol": {
        "smiles": "C1=CC(=CC=C1O)[N+](=O)[O-]",
        "Group": "Phenol",
        "pKa1": 7.15,
        "Solubility": -0.94,
        "MolLogP": 1.91,
    },
    "4-nonylphenol": {
        "smiles": "CCCCCCCCCC1=CC=C(C=C1)O",
        "Group": "Phenol",
        "pKa1": 10.28,
        "Solubility": -4.498027,
        "MolLogP": 5.76,
    },
    # --- 9. MULTI-FUNCTIONAL / COMPLEX ---
    "amoxicillin": {
        "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
        "Group": "Multi-functional",
        "pKa1": 2.40,
        "Solubility": -2.17,
        "MolLogP": 0.87,
    },
    "cysteine": {
        "smiles": "C(C(C(=O)O)N)S",
        "Group": "Multi-functional",
        "pKa1": 1.96,
        "Solubility": 0.3597,
        "MolLogP": -2.49,
    },
    "beta-alanine": {
        "smiles": "NCCC(=O)O",
        "Group": "Multi-functional",
        "pKa1": 3.55,
        "Solubility": 0.786548,
        "MolLogP": -3.07,
    },
    "3-aminobenzoic acid": {
        "smiles": "NC1=CC=CC(=C1)C(=O)O",
        "Group": "Multi-functional",
        "pKa1": 3.12,
        "Solubility": -1.3663,
        "MolLogP": 0.37,
    },
    "3-hydroxytyramine": {
        "smiles": "C1=CC(=C(C=C1CCN)O)O",
        "Group": "Multi-functional",
        "pKa1": 8.93,
        "Solubility": 0.12,
        "MolLogP": -0.98,
    },
}

# ==========================================
# 4. HELPER & INFERENCE ENGINE (FORWARD CHAINING)
# ==========================================
def is_pure_organic(smiles):
    if not smiles or "." in smiles:
        return False
    metals = ["Na", "K", "Ca", "Mg", "Fe", "Zn", "Li", "Ba"]
    for m in metals:
        if m in smiles:
            return False
    return True


def forward_chaining_engine(data, target_pH):
    logs = []
    compound_name = data.get("name", "Target")
    group = data.get("Group", "Unknown")
    pKa1 = data.get("pKa1")
    mollogp = data.get("MolLogP")
    sol_log = data.get("Solubility", 0.0)

    # Konversi Solubility dari skala log ke S0 (skala linear)
    S0 = 10**sol_log if sol_log is not None else 0.0
    smiles = data.get("smiles", "")

    logs.append(f"⚙️ [INIT] Evaluasi Senyawa: {compound_name}")
    logs.append(f"   ↳ Group: {group} | pKa1: {pKa1} | MolLogP: {mollogp} | Intrinsic log(S): {sol_log}")

    if not is_pure_organic(smiles):
        logs.append("❌ [FILTER REJECTED] Senyawa bukan molekul organik murni.")
        return 0.0, "Rejected", "#ef4444", logs

    logs.append("✅ [FILTER PASSED] Molekul Organik Murni Terdeteksi.")

    # Rule Execution berdasarkan pKa1, MolLogP, dan Group
    if pKa1 is None:
        logs.append(f"🔹 [RULE JALUR LIPOFILISITAS FIRED] pKa1 = None → Evaluasi berbasis MolLogP ({mollogp}).")
        S_total = S0
        if mollogp is not None and mollogp < 0.0:
            logs.append("   ↳ MolLogP < 0.0 → Kelarutan bawaan Tinggi.")
        elif mollogp is not None and mollogp <= 2.0:
            logs.append("   ↳ 0.0 <= MolLogP <= 2.0 → Kelarutan bawaan Sedang.")
        else:
            logs.append("   ↳ MolLogP > 2.0 → Kelarutan bawaan Rendah (Lipofilik).")

    elif group == "Carboxylic Acid" or (group == "Multi-functional" and pKa1 < 7):
        logs.append(f"🔹 [RULE ASAM LEMAH FIRED] IF Group == '{group}' DAN pKa1 ({pKa1}) DAN pH ({target_pH})")
        if target_pH > pKa1:
            logs.append(f"   ↳ pH ({target_pH}) > pKa1 ({pKa1}) → Form terionisasi (A-) mendominasi → Kelarutan Meningkat.")
        else:
            logs.append(f"   ↳ pH ({target_pH}) <= pKa1 ({pKa1}) → Form netral HA mendominasi.")
        S_total = S0 * (1 + 10 ** (target_pH - pKa1))

    elif group == "Amine":
        logs.append(f"🔹 [RULE BASA LEMAH FIRED] IF Group == 'Amine' DAN pKa1 ({pKa1}) DAN pH ({target_pH})")
        if target_pH < pKa1:
            logs.append(f"   ↳ pH ({target_pH}) < pKa1 ({pKa1}) → Terprotonasi (BH+) mendominasi → Kelarutan Meningkat.")
        else:
            logs.append(f"   ↳ pH ({target_pH}) >= pKa1 ({pKa1}) → Form netral B mendominasi.")
        S_total = S0 * (1 + 10 ** (pKa1 - target_pH))

    elif group == "Phenol":
        logs.append(f"🔹 [RULE FENOL FIRED] IF Group == 'Phenol' DAN pKa1 ({pKa1}) DAN pH ({target_pH})")
        if target_pH > pKa1:
            logs.append(f"   ↳ pH ({target_pH}) > pKa1 ({pKa1}) → Anion fenolat terbentuk → Kelarutan Meningkat.")
        else:
            logs.append(f"   ↳ pH ({target_pH}) <= pKa1 ({pKa1}) → Bentuk netral mendominasi.")
        S_total = S0 * (1 + 10 ** (target_pH - pKa1))

    else:
        logs.append(f"🔹 [RULE NETRAL / HIGH pKa FIRED] IF Group == '{group}' (pKa1 = {pKa1}) → Tidak terionisasi signifikan pada pH normal.")
        S_total = S0

    # Klasifikasi Tingkat Kelarutan (Skala Standard USP)
    if S_total >= 100.0:
        cat, color = "Sangat Mudah Larut", "#16a34a"
    elif S_total >= 30.0:
        cat, color = "Mudah Larut", "#65a30d"
    elif S_total >= 10.0:
        cat, color = "Larut", "#d97706"
    elif S_total >= 1.0:
        cat, color = "Agak Sukar Larut", "#ea580c"
    elif S_total >= 0.1:
        cat, color = "Sukar Larut", "#dc2626"
    else:
        cat, color = "Praktis Tidak Larut", "#991b1b"

    logs.append(f"🏁 [FINAL VERDICT] Kelarutan Total S_total = {S_total:.3f} mg/mL → Tingkat Kelarutan: {cat}")
    return S_total, cat, color, logs


# ==========================================
# 5. HEADER UTAMA SOLUCHAIN
# ==========================================
st.markdown(
    """
    <div class="main-header">
        <h1>🧪 SoluChain</h1>
        <p>Expert System for Predicting Organic Compound Solubility via Forward Chaining Inference</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 6. SIDEBAR INPUT & INTERAKSI HALAMAN
# ==========================================
st.sidebar.header("⚙️ Menu Navigasi & Input")

compound_options = ["-- Pilih Senyawa untuk Memulai --"] + list(
    AQSOL_IUPAC_DATABASE.keys()
)
selected_compound = st.sidebar.selectbox(
    "Pilih Senyawa Organik:", compound_options
)

# ==========================================
# 7. LOGIKA TAMPILAN: LANDING PAGE vs ANALISIS SENYAWA
# ==========================================
if selected_compound == "-- Pilih Senyawa untuk Memulai --":
    st.markdown(
        """
        <div class="hero-card">
            <h2 style="color: #1e3a8a; font-weight: 700;">Selamat Datang di SoluChain! 👋</h2>
            <p style="color: #475569; font-size: 1.1rem; max-width: 680px; margin: 12px auto 24px auto;">
                Sistem Pakar Berbasis Pengetahuan (<i>Knowledge-Based System</i>) untuk memprediksi dan menjelaskan 
                <b>kelarutan senyawa organik dalam air</b> berdasarkan ionisasi asam-basa dan parameter MolLogP.
            </p>
            <div style="background-color: #eff6ff; padding: 16px; border-radius: 8px; display: inline-block; text-align: left; border: 1px solid #bfdbfe;">
                <b style="color: #1e40af;">📌 Cara Menggunakan Aplikasi:</b>
                <ol style="color: #1e3a8a; margin-top: 8px; margin-bottom: 0; padding-left: 20px;">
                    <li>Buka menu di sebelah kiri (<b>Sidebar</b>).</li>
                    <li>Pilih salah satu <b>Senyawa Organik</b> dari daftar dropdown.</li>
                    <li>Atur nilai <b>pH Lingkungan Pelarut</b> sesuai kebutuhan.</li>
                    <li>Sistem akan secara otomatis menjalankan mesin inferensi <i>Forward Chaining</i>!</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📚 Dataset Gabungan AqSolDB + IUPAC pKa (45 Senyawa)")
    
    # Menampilkan DataFrame dengan Judul Kolom 'Compound Name' di Atas Index
    df_db_preview = pd.DataFrame(AQSOL_IUPAC_DATABASE).T
    df_db_preview.index.name = "Compound Name"
    st.dataframe(df_db_preview, use_container_width=True)

else:
    compound_data = AQSOL_IUPAC_DATABASE[selected_compound]
    compound_data["name"] = selected_compound

    target_pH = st.sidebar.slider(
        "Atur pH Environment Pelarut:", 1.0, 14.0, 7.4, step=0.1
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Physical Descriptors")
    st.sidebar.text(f"SMILES: {compound_data.get('smiles', 'N/A')}")
    st.sidebar.text(f"Group: {compound_data.get('Group', 'N/A')}")
    st.sidebar.text(f"MolLogP: {compound_data.get('MolLogP', 'N/A')}")
    st.sidebar.text(f"Solubility (log): {compound_data.get('Solubility', 'N/A')}")

    # Jalankan Engine Inferensi
    S_total, category, cat_color, rule_logs = forward_chaining_engine(
        compound_data, target_pH
    )

    # Display KPI Cards Berwarna
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">TOTAL KELARUTAN</div>
            <div class="metric-value">{S_total:.3f} <span style="font-size:12px">mg/mL</span></div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">GUGUS FUNGSI (GROUP)</div>
            <div class="metric-value" style="color:#2563eb">{compound_data.get('Group', 'Unknown')}</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">TINGKAT KELARUTAN</div>
            <div class="metric-value" style="color:{cat_color}; font-size:18px">{category}</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        pka1_val = compound_data.get("pKa1")
        pka_display = f"{pka1_val}" if pka1_val is not None else "N/A"

        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">pKa1 (IUPAC)</div>
            <div class="metric-value" style="color:#d97706">{pka_display}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs Output Navigation
    tab1, tab2, tab3 = st.tabs(
        [
            "🧠 Traceable Reasoning Chain (Forward Chaining)",
            "📈 Profil pH vs Kelarutan",
            "📊 Dataset Table",
        ]
    )

    with tab1:
        st.subheader("Explainable AI: Rule Execution Log")
        st.info(
            "Alur aturan (*Forward Chaining Rules*) yang dieksekusi secara transparan berdasarkan sifat fisikokimia molekul:"
        )
        for log in rule_logs:
            st.markdown(
                f'<div class="rule-box">{log}</div>', unsafe_allow_html=True
            )

    with tab2:
        st.subheader("Profil pH-Kelarutan (Henderson-Hasselbalch Curve)")

        pH_array = np.linspace(1, 14, 200)
        S_array = [
            forward_chaining_engine(compound_data, ph)[0] for ph in pH_array
        ]

        fig = go.Figure()

        # Garis Grafik Utama
        fig.add_trace(
            go.Scatter(
                x=pH_array,
                y=S_array,
                mode="lines",
                name="Profil Kelarutan",
                line=dict(color="#2563eb", width=3),
            )
        )

        # Titik Merah State Saat Ini
        fig.add_trace(
            go.Scatter(
                x=[target_pH],
                y=[S_total],
                mode="markers+text",
                name=f"State Saat Ini (pH {target_pH})",
                marker=dict(color="#dc2626", size=12, symbol="diamond"),
                text=[f"  {S_total:.2f} mg/mL"],
                textposition="top right",
                textfont=dict(color="#1e3a8a", size=13),
            )
        )

        # Pengaturan Kontras Warna Teks & Sumbu
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(color="#0f172a", size=13),
            xaxis=dict(
                title="<b>pH Larutan Environment</b>",
                title_font=dict(color="#1e3a8a", size=14),
                tickfont=dict(color="#0f172a", size=12),
                gridcolor="#cbd5e1",
                linecolor="#94a3b8",
            ),
            yaxis=dict(
                title="<b>Total Kelarutan (mg/mL) - Skala Log</b>",
                title_font=dict(color="#1e3a8a", size=14),
                tickfont=dict(color="#0f172a", size=12),
                gridcolor="#cbd5e1",
                linecolor="#94a3b8",
                type="log",
            ),
            legend=dict(
                font=dict(color="#0f172a", size=12),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#cbd5e1",
                borderwidth=1,
            ),
            height=440,
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Data AqSolDB + IUPAC Joined Table")
        df_db = pd.DataFrame(AQSOL_IUPAC_DATABASE).T
        df_db.index.name = "Compound Name"
        st.dataframe(df_db, use_container_width=True)
