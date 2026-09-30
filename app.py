import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(
    page_title="SoluBayes - AI Organic Compound Solubility Predictor",
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
    
    /* Header Judul Utama SoluBayes */
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

    /* Kartu Metrik Berwarna Biru-Putih */
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
        font-size: 22px; 
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

    /* Kotak Rule Execution Log */
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

    /* Hero Banner Halaman Awal */
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
# DATABASE SENYAWA LENGKAP (45 SENYAWA + SMILES)
# =========================================================
AQSOL_IUPAC_DATABASE = {
    # --- ALCOHOL ---
    "butan-1-ol": {
        "smiles": "CCCCO",
        "type": "Neutral",
        "fg": ["alcohol"],
        "pKa1": 16.1,
        "pKa2": None,
        "S0": -0.050409,
        "logP": None,
    },
    "benzyl alcohol": {
        "smiles": "C1=CC=C(C=C1)CO",
        "type": "Neutral",
        "fg": ["alcohol", "aromatic"],
        "pKa1": 15.4,
        "pKa2": None,
        "S0": -0.4319,
        "logP": None,
    },
    "1-propanol": {
        "smiles": "CCCO",
        "type": "Neutral",
        "fg": ["alcohol"],
        "pKa1": 16.1,
        "pKa2": None,
        "S0": 0.62,
        "logP": None,
    },
    "ethylene glycol": {
        "smiles": "OCCO",
        "type": "Neutral",
        "fg": ["alcohol"],
        "pKa1": 15.4,
        "pKa2": None,
        "S0": 1.2071,
        "logP": None,
    },
    "2-butanol": {
        "smiles": "CCC(C)O",
        "type": "Neutral",
        "fg": ["alcohol"],
        "pKa1": 17.6,
        "pKa2": None,
        "S0": 0.3877,
        "logP": None,
    },
    # --- CARBOXYLIC ACID ---
    "propanoic acid": {
        "smiles": "CCC(=O)O",
        "type": "Weak Acid",
        "fg": ["carboxylic_acid"],
        "pKa1": 4.86,
        "pKa2": None,
        "S0": 1.130305,
        "logP": None,
    },
    "acetic acid": {
        "smiles": "CC(=O)O",
        "type": "Weak Acid",
        "fg": ["carboxylic_acid"],
        "pKa1": 4.73,
        "pKa2": None,
        "S0": 1.001718,
        "logP": None,
    },
    "benzoic acid": {
        "smiles": "C1=CC=C(C=C1)C(=O)O",
        "type": "Weak Acid",
        "fg": ["carboxylic_acid", "aromatic"],
        "pKa1": 4.13,
        "pKa2": None,
        "S0": -1.61993,
        "logP": None,
    },
    "cholic acid": {
        "smiles": "CC(CCC(=O)O)C1CCC2C1(CCC3C2C(CC4C3(CCC(C4)O)C)O)O",
        "type": "Weak Acid",
        "fg": ["carboxylic_acid", "alcohol"],
        "pKa1": 4.98,
        "pKa2": None,
        "S0": -3.3682,
        "logP": None,
    },
    "cumic acid": {
        "smiles": "CC(C)C1=CC=C(C=C1)C(=O)O",
        "type": "Weak Acid",
        "fg": ["carboxylic_acid", "aromatic"],
        "pKa1": 4.34,
        "pKa2": None,
        "S0": -3.0364,
        "logP": None,
    },
    # --- ETHER ---
    "allyl ether": {
        "smiles": "C=CCOCC=C",
        "type": "Neutral",
        "fg": ["ether", "alkene"],
        "pKa1": None,
        "pKa2": None,
        "S0": -0.0206,
        "logP": 1.36,
    },
    "butoxybenzene": {
        "smiles": "CCCCOC1=CC=CC=C1",
        "type": "Neutral",
        "fg": ["ether", "aromatic"],
        "pKa1": None,
        "pKa2": None,
        "S0": -3.61325,
        "logP": 3.07,
    },
    "methoxychlor": {
        "smiles": "COC1=CC=C(C=C1)C(C2=CC=C(C=C2)OC)C(Cl)(Cl)Cl",
        "type": "Neutral",
        "fg": ["ether", "aromatic", "halogen"],
        "pKa1": None,
        "pKa2": None,
        "S0": -6.5386,
        "logP": 5.78,
    },
    "diethyl ether": {
        "smiles": "CCOCC",
        "type": "Neutral",
        "fg": ["ether"],
        "pKa1": None,
        "pKa2": None,
        "S0": -0.0889,
        "logP": 0.89,
    },
    "anisole": {
        "smiles": "COC1=CC=CC=C1",
        "type": "Neutral",
        "fg": ["ether", "aromatic"],
        "pKa1": None,
        "pKa2": None,
        "S0": -1.85,
        "logP": 2.11,
    },
    # --- ESTER ---
    "dimethoate": {
        "smiles": "CNC(=O)CSP(=S)(OC)OC",
        "type": "Neutral",
        "fg": ["ester", "amide"],
        "pKa1": None,
        "pKa2": None,
        "S0": -0.9624,
        "logP": -0.08,
    },
    "cycloate": {
        "smiles": "CCN(C1CCCCC1)C(=O)SCC",
        "type": "Neutral",
        "fg": ["ester"],
        "pKa1": None,
        "pKa2": None,
        "S0": -3.4037,
        "logP": 3.11,
    },
    "fenthoate": {
        "smiles": "CCOC(=O)C(C1=CC=CC=C1)SP(=S)(OC)OC",
        "type": "Neutral",
        "fg": ["ester"],
        "pKa1": None,
        "pKa2": None,
        "S0": -4.4643,
        "logP": 3.68,
    },
    "ethyl acetate": {
        "smiles": "CCOC(C)=O",
        "type": "Neutral",
        "fg": ["ester"],
        "pKa1": None,
        "pKa2": None,
        "S0": -0.025404,
        "logP": 0.73,
    },
    "methyl benzoate": {
        "smiles": "COC(=O)C1=CC=CC=C1",
        "type": "Neutral",
        "fg": ["ester", "aromatic"],
        "pKa1": None,
        "pKa2": None,
        "S0": -1.811798,
        "logP": 2.12,
    },
    # --- ALDEHYDE ---
    "formaldehyde": {
        "smiles": "C=O",
        "type": "Neutral",
        "fg": ["aldehyde"],
        "pKa1": 12.2,
        "pKa2": None,
        "S0": 1.1206,
        "logP": None,
    },
    "acetaldehyde": {
        "smiles": "CC=O",
        "type": "Neutral",
        "fg": ["aldehyde"],
        "pKa1": 13.57,
        "pKa2": None,
        "S0": 1.3561,
        "logP": None,
    },
    "benzaldehyde": {
        "smiles": "C1=CC=C(C=C1)C=O",
        "type": "Neutral",
        "fg": ["aldehyde", "aromatic"],
        "pKa1": 14.9,
        "pKa2": None,
        "S0": -1.209572,
        "logP": None,
    },
    "4-methylbenzaldehyde": {
        "smiles": "CC1=CC=C(C=C1)C=O",
        "type": "Neutral",
        "fg": ["aldehyde", "aromatic"],
        "pKa1": 15.39,
        "pKa2": None,
        "S0": -1.723702,
        "logP": None,
    },
    "2,2,2-trichloroacetaldehyde": {
        "smiles": "C(=O)C(Cl)(Cl)Cl",
        "type": "Neutral",
        "fg": ["aldehyde", "halogen"],
        "pKa1": 9.95,
        "pKa2": None,
        "S0": -0.691341,
        "logP": None,
    },
    # --- KETONE ---
    "cyclohexanone": {
        "smiles": "O=C1CCCCC1",
        "type": "Neutral",
        "fg": ["ketone"],
        "pKa1": 16.7,
        "pKa2": None,
        "S0": -0.05737,
        "logP": None,
    },
    "4-methylpent-3-en-2-one": {
        "smiles": "CC(=CC(=O)C)C",
        "type": "Neutral",
        "fg": ["ketone", "alkene"],
        "pKa1": 20.5,
        "pKa2": None,
        "S0": -0.56,
        "logP": None,
    },
    "1-methylpyrrolidin-2-one": {
        "smiles": "CN1CCCC1=O",
        "type": "Neutral",
        "fg": ["ketone", "amide"],
        "pKa1": 24.0,
        "pKa2": None,
        "S0": 1.00,
        "logP": None,
    },
    "acetone": {
        "smiles": "CC(=O)C",
        "type": "Neutral",
        "fg": ["ketone"],
        "pKa1": 20.0,
        "pKa2": None,
        "S0": 1.236,
        "logP": None,
    },
    "1-phenylethan-1-one": {
        "smiles": "CC(=O)C1=CC=CC=C1",
        "type": "Neutral",
        "fg": ["ketone", "aromatic"],
        "pKa1": 19.2,
        "pKa2": None,
        "S0": -1.280387,
        "logP": None,
    },
    # --- AMINE ---
    "ethylamine": {
        "smiles": "CCN",
        "type": "Weak Base",
        "fg": ["amine"],
        "pKa1": 10.79,
        "pKa2": None,
        "S0": 1.346,
        "logP": None,
    },
    "hexylamine": {
        "smiles": "CCCCCCN",
        "type": "Weak Base",
        "fg": ["amine"],
        "pKa1": 10.56,
        "pKa2": None,
        "S0": -1.1,
        "logP": None,
    },
    "aniline": {
        "smiles": "C1=CC=C(C=C1)N",
        "type": "Weak Base",
        "fg": ["amine", "aromatic"],
        "pKa1": 4.60,
        "pKa2": None,
        "S0": -0.425017,
        "logP": None,
    },
    "pyridine": {
        "smiles": "C1=CC=NC=C1",
        "type": "Weak Base",
        "fg": ["amine", "aromatic"],
        "pKa1": 5.25,
        "pKa2": None,
        "S0": 0.76,
        "logP": None,
    },
    "benzylamine": {
        "smiles": "C1=CC=C(C=C1)CN",
        "type": "Weak Base",
        "fg": ["amine", "aromatic"],
        "pKa1": 9.33,
        "pKa2": None,
        "S0": -1.53,
        "logP": None,
    },
    # --- PHENOL ---
    "phenol": {
        "smiles": "C1=CC=C(C=C1)O",
        "type": "Phenol",
        "fg": ["phenol", "aromatic"],
        "pKa1": 9.98,
        "pKa2": None,
        "S0": -0.04,
        "logP": None,
    },
    "4-methylphenol": {
        "smiles": "CC1=CC=C(C=C1)O",
        "type": "Phenol",
        "fg": ["phenol", "aromatic"],
        "pKa1": 10.26,
        "pKa2": None,
        "S0": -0.701548,
        "logP": None,
    },
    "o-aminophenol": {
        "smiles": "C1=CC=C(C(=C1)N)O",
        "type": "Amphoteric",
        "fg": ["phenol", "amine", "aromatic"],
        "pKa1": 9.71,
        "pKa2": None,
        "S0": -0.7369,
        "logP": None,
    },
    "4-nitrophenol": {
        "smiles": "C1=CC(=CC=C1O)[N+](=O)[O-]",
        "type": "Phenol",
        "fg": ["phenol", "aromatic", "nitro"],
        "pKa1": 7.15,
        "pKa2": None,
        "S0": -0.94,
        "logP": None,
    },
    "4-nonylphenol": {
        "smiles": "CCCCCCCCCC1=CC=C(C=C1)O",
        "type": "Phenol",
        "fg": ["phenol", "aromatic"],
        "pKa1": 10.28,
        "pKa2": None,
        "S0": -4.498027,
        "logP": None,
    },
    # --- MULTI-FUNCTIONAL / COMPLEX ---
    "amoxicillin": {
        "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
        "type": "Amphoteric",
        "fg": ["carboxylic_acid", "amine", "phenol"],
        "pKa1": 2.40,
        "pKa2": 9.6,
        "S0": -2.17,
        "logP": None,
    },
    "cysteine": {
        "smiles": "C(C(C(=O)O)N)S",
        "type": "Amphoteric",
        "fg": ["carboxylic_acid", "amine"],
        "pKa1": 1.96,
        "pKa2": 8.18,
        "S0": 0.3597,
        "logP": None,
    },
    "beta-alanine": {
        "smiles": "NCCC(=O)O",
        "type": "Amphoteric",
        "fg": ["carboxylic_acid", "amine"],
        "pKa1": 3.55,
        "pKa2": 10.24,
        "S0": 0.786548,
        "logP": None,
    },
    "3-aminobenzoic acid": {
        "smiles": "NC1=CC=CC(=C1)C(=O)O",
        "type": "Amphoteric",
        "fg": ["carboxylic_acid", "amine", "aromatic"],
        "pKa1": 3.12,
        "pKa2": 4.74,
        "S0": -1.3663,
        "logP": None,
    },
    "3-hydroxytyramine": {
        "smiles": "C1=CC(=C(C=C1CCN)O)O",
        "type": "Amphoteric",
        "fg": ["phenol", "amine", "aromatic"],
        "pKa1": 8.93,
        "pKa2": 10.6,
        "S0": 0.12,
        "logP": None,
    },
}


# ==========================================
# 4. HELPER & INFERENCE ENGINE (5 LAPIS)
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
    smiles = data.get("smiles", "")

    logs.append(f"⚙️ [INIT] Evaluasi Senyawa: {data.get('name', 'Target')}")
    logs.append(f"   ↳ Structure SMILES: {smiles}")

    if not is_pure_organic(smiles):
        logs.append("❌ [FILTER REJECTED] Senyawa bukan molekul organik murni.")
        return 0.0, "Rejected", "#ef4444", logs

    logs.append("✅ [FILTER PASSED] Molekul Organik Murni Terdeteksi.")

    c_type = data.get("type")
    pKa1 = data.get("pKa1")
    pKa2 = data.get("pKa2")
    
    # Konversi S0 dari log scale (AqSolDB) ke linear scale
    S0_log = data.get("S0", 0.0)
    S0 = 10 ** S0_log if S0_log is not None else 0.0
    
    fg_list = data.get("fg", [])

    # RULE C: Neutral / Non-Ionizable
    if c_type == "Neutral" or pKa1 is None:
        logs.append(f"🔹 [RULE C FIRED] IF FunctionalGroup ∈ {fg_list} → Non-Ionizable / Independent pH.")
        logs.append(f"   ↳ Kelarutan ditentukan oleh parameter fisik: LogP={data.get('logP', 'N/A')}, TPSA={data.get('tpsa', 'N/A')}.")
        S_total = S0

    # RULE B: Asam / Basa Kuat
    elif c_type in ["Strong Acid", "Strong Base"]:
        logs.append(f"🔹 [RULE B FIRED] IF Type == '{c_type}' → Terionisasi Sempurna (100%).")
        S_total = S0 * 100.0

    # RULE D: Fenol
    elif c_type == "Phenol":
        logs.append(f"🔹 [RULE D FIRED] IF FunctionalGroup == 'phenol' DAN pKa ({pKa1}) ~ 9-10")
        logs.append("   ↳ Anion fenoksida distabilkan oleh resonansi cincin aromatik.")
        if target_pH > pKa1:
            logs.append(f"   ↳ pH ({target_pH}) > pKa ({pKa1}) → Terbentuk anion fenolat → Kelarutan Tinggi.")
        else:
            logs.append(f"   ↳ pH ({target_pH}) < pKa ({pKa1}) → Bentuk molekul netral mendominasi.")
        S_total = S0 * (1 + 10 ** (target_pH - pKa1))

    # RULE E: Senyawa Amfoter / Zwitterion
    elif c_type == "Amphoteric":
        logs.append("🔹 [RULE E FIRED] IF Gugus Amina DAN Asam Karboksilat Ada → Senyawa Amfoter.")
        pKa2_val = pKa2 if pKa2 is not None else pKa1
        pI = (pKa1 + pKa2_val) / 2
        logs.append(f"   ↳ Titik Isoelektrik (pI) = {pI:.2f}.")
        if abs(target_pH - pI) < 0.5:
            logs.append(f"   ↳ Pada pH {target_pH} (~pI): Dominan Zwitterion → Kelarutan Minimum.")
        elif target_pH < pKa1:
            logs.append(f"   ↳ Pada pH {target_pH} (< pKa1): Spesies Kationik mendominasi (+1).")
        else:
            logs.append(f"   ↳ Pada pH {target_pH} (> pKa2_val): Spesies Anionik mendominasi (-1).")
        S_total = S0 * (1 + 10 ** (target_pH - pKa2_val) + 10 ** (pKa1 - target_pH))

    # RULE A: Asam / Basa Lemah
    elif c_type == "Weak Acid":
        logs.append(f"🔹 [RULE A FIRED] IF Asam Lemah DAN pKa ({pKa1}) DAN pH ({target_pH})")
        if target_pH > pKa1:
            logs.append(f"   ↳ pH ({target_pH}) > pKa ({pKa1}) → Form terionisasi (A-) mendominasi.")
        else:
            logs.append(f"   ↳ pH ({target_pH}) < pKa ({pKa1}) → Form netral HA mendominasi.")
        S_total = S0 * (1 + 10 ** (target_pH - pKa1))

    elif c_type == "Weak Base":
        logs.append(f"🔹 [RULE A FIRED] IF Basa Lemah DAN pKa ({pKa1}) DAN pH ({target_pH})")
        if target_pH < pKa1:
            logs.append(f"   ↳ pH ({target_pH}) < pKa ({pKa1}) → Terprotonasi BH+ mendominasi.")
        else:
            logs.append(f"   ↳ pH ({target_pH}) > pKa ({pKa1}) → Bentuk netral B mendominasi.")
        S_total = S0 * (1 + 10 ** (pKa1 - target_pH))

    else:
        S_total = S0

    # USP Classification
    if S_total >= 100.0:
        cat, color = "Very Soluble", "#16a34a"
    elif S_total >= 30.0:
        cat, color = "Freely Soluble", "#65a30d"
    elif S_total >= 10.0:
        cat, color = "Soluble", "#d97706"
    elif S_total >= 1.0:
        cat, color = "Sparingly Soluble", "#ea580c"
    elif S_total >= 0.1:
        cat, color = "Slightly Soluble", "#dc2626"
    else:
        cat, color = "Practically Insoluble", "#991b1b"

    logs.append(f"🏁 [FINAL VERDICT] Kelarutan Total S_total = {S_total:.3f} mg/mL → Kategori: {cat}")
    return S_total, cat, color, logs


# ==========================================
# 5. HEADER UTAMA SOLUBAYES
# ==========================================
st.markdown(
    """
    <div class="main-header">
        <h1>🧪 SoluBayes</h1>
        <p>Expert System for Predicting Organic Compound Solubility via Forward Chaining Inference</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 6. SIDEBAR INPUT & INTERAKSI HALAMAN
# ==========================================
st.sidebar.header("⚙️ Menu Navigasi & Input")

compound_options = ["-- Pilih Senyawa untuk Memulai --"] + list(AQSOL_IUPAC_DATABASE.keys())
selected_compound = st.sidebar.selectbox("Pilih Senyawa Organik:", compound_options)

# ==========================================
# 7. LOGIKA TAMPILAN: LANDING PAGE vs ANALISIS SENYAWA
# ==========================================
if selected_compound == "-- Pilih Senyawa untuk Memulai --":
    st.markdown(
        """
        <div class="hero-card">
            <h2 style="color: #1e3a8a; font-weight: 700;">Selamat Datang di SoluBayes! 👋</h2>
            <p style="color: #475569; font-size: 1.1rem; max-width: 680px; margin: 12px auto 24px auto;">
                Sistem Pakar Berbasis Pengetahuan (<i>Knowledge-Based System</i>) untuk memprediksi dan menjelaskan 
                <b>kelarutan senyawa organik dalam air</b> berdasarkan ionisasi asam-basa (<i>Rule A–E</i>).
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
    st.subheader("📚 Dataset Gabungan AqSolDB + IUPAC pKa")
    df_db_preview = pd.DataFrame(AQSOL_IUPAC_DATABASE).T
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
    st.sidebar.text(f"MolLogP: {compound_data.get('logP', 'N/A')}")
    st.sidebar.text(f"TPSA: {compound_data.get('tpsa', 'N/A')} Å²")
    st.sidebar.text(f"H-Donors: {compound_data.get('h_donors', 'N/A')}")
    st.sidebar.text(f"H-Acceptors: {compound_data.get('h_acceptors', 'N/A')}")

    # Jalankan Engine
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
            f"""<div class="metric-card"><div class="metric-label">TIPE MOLEKUL</div>
            <div class="metric-value" style="color:#2563eb">{compound_data.get('type', 'Unknown')}</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">KLASIFIKASI USP</div>
            <div class="metric-value" style="color:{cat_color}; font-size:18px">{category}</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        pka_display = f"{compound_data.get('pKa1')}" if compound_data.get("pKa1") is not None else "N/A"
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">pKa (IUPAC)</div>
            <div class="metric-value" style="color:#d97706">{pka_display}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs Output
    tab1, tab2, tab3 = st.tabs(
        ["🧠 Traceable Reasoning Chain (Forward Chaining)", "📈 Profil pH vs Kelarutan", "📊 Dataset Table"]
    )

    with tab1:
        st.subheader("Explainable AI: Rule Execution Log")
        st.info("Alur aturan (*Rules A-E*) yang dieksekusi secara transparan berdasarkan sifat fisikokimia molekul:")
        for log in rule_logs:
            st.markdown(f'<div class="rule-box">{log}</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Profil pH-Kelarutan (Henderson-Hasselbalch Curve)")

        pH_array = np.linspace(1, 14, 200)
        S_array = [forward_chaining_engine(compound_data, ph)[0] for ph in pH_array]

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
        st.dataframe(df_db, use_container_width=True)
