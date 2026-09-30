import math
import urllib.parse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
# 2. CUSTOM CSS: TEMA ELEGAN
# ==========================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 24px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .main-header p {
        color: #dbeafe;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 0;
    }

    .metric-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 4px solid #2563eb;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        min-height: 120px;
    }
    .metric-value { 
        font-size: 18px; 
        font-weight: 800; 
        color: #1e3a8a; 
    }
    .metric-label { 
        font-size: 11px; 
        color: #64748b; 
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-sub {
        font-size: 12px;
        color: #475569;
        font-weight: 600;
        margin-top: 4px;
    }
    .metric-range {
        font-size: 11px;
        color: #0369a1;
        font-weight: 700;
        margin-top: 2px;
        background-color: #f0f9ff;
        padding: 2px 6px;
        border-radius: 4px;
        display: inline-block;
    }

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
        "Smiles": "CCCCO",
        "Group": "Alcohol",
        "pKa1": 16.1,
        "Log S0": -0.050409,
        "MolLogP": 0.88,
    },
    "benzyl alcohol": {
        "Smiles": "C1=CC=C(C=C1)CO",
        "Group": "Alcohol",
        "pKa1": 15.4,
        "Log S0": -0.4319,
        "MolLogP": 1.10,
    },
    "1-propanol": {
        "Smiles": "CCCO",
        "Group": "Alcohol",
        "pKa1": 16.1,
        "Log S0": 0.62,
        "MolLogP": 0.25,
    },
    "ethylene glycol": {
        "Smiles": "OCCO",
        "Group": "Alcohol",
        "pKa1": 15.4,
        "Log S0": 1.2071,
        "MolLogP": -1.36,
    },
    "2-butanol": {
        "Smiles": "CCC(C)O",
        "Group": "Alcohol",
        "pKa1": 17.6,
        "Log S0": 0.3877,
        "MolLogP": 0.65,
    },
    # --- 2. CARBOXYLIC ACID ---
    "propanoic acid": {
        "Smiles": "CCC(=O)O",
        "Group": "Carboxylic Acid",
        "pKa1": 4.86,
        "Log S0": 1.130305,
        "MolLogP": 0.33,
    },
    "acetic acid": {
        "Smiles": "CC(=O)O",
        "Group": "Carboxylic Acid",
        "pKa1": 4.73,
        "Log S0": 1.001718,
        "MolLogP": -0.17,
    },
    "benzoic acid": {
        "Smiles": "C1=CC=C(C=C1)C(=O)O",
        "Group": "Carboxylic Acid",
        "pKa1": 4.13,
        "Log S0": -1.61993,
        "MolLogP": 1.87,
    },
    "cholic acid": {
        "Smiles": "C[C@H](CCC(=O)O)[C@H]1CC[C@@H]2[C@@]1([C@H](C[C@H]3[C@H]2[C@@H](C[C@H]4[C@@]3(CC[C@H](C4)O)C)O)O)C",
        "Group": "Carboxylic Acid",
        "pKa1": 4.98,
        "Log S0": -3.3682,
        "MolLogP": 2.02,
    },
    "cumic acid": {
        "Smiles": "CC(C)C1=CC=C(C=C1)C(=O)O",
        "Group": "Carboxylic Acid",
        "pKa1": 4.34,
        "Log S0": -3.0364,
        "MolLogP": 2.91,
    },
    # --- 3. ETHER ---
    "allyl ether": {
        "Smiles": "C=CCOCC=C",
        "Group": "Ether",
        "pKa1": None,
        "Log S0": -0.0206,
        "MolLogP": 1.36,
    },
    "butoxybenzene": {
        "Smiles": "CCCCOC1=CC=CC=C1",
        "Group": "Ether",
        "pKa1": None,
        "Log S0": -3.61325,
        "MolLogP": 3.07,
    },
    "methoxychlor": {
        "Smiles": "COC1=CC=C(C=C1)C(C2=CC=C(C=C2)OC)C(Cl)(Cl)Cl",
        "Group": "Ether",
        "pKa1": None,
        "Log S0": -6.5386,
        "MolLogP": 5.78,
    },
    "diethyl ether": {
        "Smiles": "CCOCC",
        "Group": "Ether",
        "pKa1": None,
        "Log S0": -0.0889,
        "MolLogP": 0.89,
    },
    "anisole": {
        "Smiles": "COC1=CC=CC=C1",
        "Group": "Ether",
        "pKa1": None,
        "Log S0": -1.85,
        "MolLogP": 2.11,
    },
    # --- 4. ESTER ---
    "dimethoate": {
        "Smiles": "CNC(=O)CSP(=S)(OC)OC",
        "Group": "Ester",
        "pKa1": None,
        "Log S0": -0.9624,
        "MolLogP": -0.08,
    },
    "cycloate": {
        "Smiles": "CCN(C1CCCCC1)C(=O)SCC",
        "Group": "Ester",
        "pKa1": None,
        "Log S0": -3.4037,
        "MolLogP": 3.11,
    },
    "fenthoate": {
        "Smiles": "CCOC(=O)C(C1=CC=CC=C1)SP(=S)(OC)OC",
        "Group": "Ester",
        "pKa1": None,
        "Log S0": -4.4643,
        "MolLogP": 3.68,
    },
    "ethyl acetate": {
        "Smiles": "CCOC(C)=O",
        "Group": "Ester",
        "pKa1": None,
        "Log S0": -0.025404,
        "MolLogP": 0.73,
    },
    "methyl benzoate": {
        "Smiles": "COC(=O)C1=CC=CC=C1",
        "Group": "Ester",
        "pKa1": None,
        "Log S0": -1.811798,
        "MolLogP": 2.12,
    },
    # --- 5. ALDEHYDE ---
    "formaldehyde": {
        "Smiles": "C=O",
        "Group": "Aldehyde",
        "pKa1": 12.2,
        "Log S0": 1.1206,
        "MolLogP": 0.35,
    },
    "acetaldehyde": {
        "Smiles": "CC=O",
        "Group": "Aldehyde",
        "pKa1": 13.57,
        "Log S0": 1.3561,
        "MolLogP": -0.17,
    },
    "benzaldehyde": {
        "Smiles": "C1=CC=C(C=C1)C=O",
        "Group": "Aldehyde",
        "pKa1": 14.9,
        "Log S0": -1.209572,
        "MolLogP": 1.48,
    },
    "4-methylbenzaldehyde": {
        "Smiles": "CC1=CC=C(C=C1)C=O",
        "Group": "Aldehyde",
        "pKa1": 15.39,
        "Log S0": -1.723702,
        "MolLogP": 2.01,
    },
    "2,2,2-trichloroacetaldehyde": {
        "Smiles": "C(=O)C(Cl)(Cl)Cl",
        "Group": "Aldehyde",
        "pKa1": 9.95,
        "Log S0": -0.691341,
        "MolLogP": 1.42,
    },
    # --- 6. KETONE ---
    "cyclohexanone": {
        "Smiles": "O=C1CCCCC1",
        "Group": "Ketone",
        "pKa1": 16.7,
        "Log S0": -0.05737,
        "MolLogP": 0.81,
    },
    "4-methylpent-3-en-2-one": {
        "Smiles": "CC(=CC(=O)C)C",
        "Group": "Ketone",
        "pKa1": 20.5,
        "Log S0": -0.56,
        "MolLogP": 1.33,
    },
    "1-methylpyrrolidin-2-one": {
        "Smiles": "CN1CCCC1=O",
        "Group": "Ketone",
        "pKa1": 24.0,
        "Log S0": 1.00,
        "MolLogP": -0.38,
    },
    "acetone": {
        "Smiles": "CC(=O)C",
        "Group": "Ketone",
        "pKa1": 20.0,
        "Log S0": 1.236,
        "MolLogP": -0.24,
    },
    "1-phenylethan-1-one": {
        "Smiles": "CC(=O)C1=CC=CC=C1",
        "Group": "Ketone",
        "pKa1": 19.2,
        "Log S0": -1.280387,
        "MolLogP": 1.58,
    },
    # --- 7. AMINE ---
    "ethylamine": {
        "Smiles": "CCN",
        "Group": "Amine",
        "pKa1": 10.79,
        "Log S0": 1.346,
        "MolLogP": -0.13,
    },
    "hexylamine": {
        "Smiles": "CCCCCCN",
        "Group": "Amine",
        "pKa1": 10.56,
        "Log S0": -1.1,
        "MolLogP": 2.06,
    },
    "aniline": {
        "Smiles": "C1=CC=C(C=C1)N",
        "Group": "Amine",
        "pKa1": 4.60,
        "Log S0": -0.425017,
        "MolLogP": 0.90,
    },
    "pyridine": {
        "Smiles": "C1=CC=NC=C1",
        "Group": "Amine",
        "pKa1": 5.25,
        "Log S0": 0.76,
        "MolLogP": 0.65,
    },
    "benzylamine": {
        "Smiles": "C1=CC=C(C=C1)CN",
        "Group": "Amine",
        "pKa1": 9.33,
        "Log S0": -1.53,
        "MolLogP": 1.09,
    },
    # --- 8. PHENOL ---
    "phenol": {
        "Smiles": "C1=CC=C(C=C1)O",
        "Group": "Phenol",
        "pKa1": 9.98,
        "Log S0": -0.04,
        "MolLogP": 1.46,
    },
    "4-methylphenol": {
        "Smiles": "CC1=CC=C(C=C1)O",
        "Group": "Phenol",
        "pKa1": 10.26,
        "Log S0": -0.701548,
        "MolLogP": 1.94,
    },
    "o-aminophenol": {
        "Smiles": "C1=CC=C(C(=C1)N)O",
        "Group": "Phenol",
        "pKa1": 9.71,
        "Log S0": -0.7369,
        "MolLogP": 0.62,
    },
    "4-nitrophenol": {
        "Smiles": "C1=CC(=CC=C1O)[N+](=O)[O-]",
        "Group": "Phenol",
        "pKa1": 7.15,
        "Log S0": -0.94,
        "MolLogP": 1.91,
    },
    "4-nonylphenol": {
        "Smiles": "CCCCCCCCC1=CC=C(C=C1)O",
        "Group": "Phenol",
        "pKa1": 10.28,
        "Log S0": -4.498027,
        "MolLogP": 5.76,
    },
    # --- 9. MULTI-FUNCTIONAL / COMPLEX ---
    "amoxicillin": {
        "Smiles": "CC1([C@@H](N2[C@@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C",
        "Group": "Multi-functional",
        "pKa1": 2.40,
        "Log S0": -2.17,
        "MolLogP": 0.87,
    },
    "cysteine": {
        "Smiles": "C([C@@H](C(=O)O)N)S",
        "Group": "Multi-functional",
        "pKa1": 1.96,
        "Log S0": 0.3597,
        "MolLogP": -2.49,
    },
    "beta-alanine": {
        "Smiles": "NCCC(=O)O",
        "Group": "Multi-functional",
        "pKa1": 3.55,
        "Log S0": 0.786548,
        "MolLogP": -3.07,
    },
    "3-aminobenzoic acid": {
        "Smiles": "NC1=CC=CC(=C1)C(=O)O",
        "Group": "Multi-functional",
        "pKa1": 3.12,
        "Log S0": -1.3663,
        "MolLogP": 0.37,
    },
    "3-hydroxytyramine": {
        "Smiles": "C1=CC(=C(C=C1CCN)O)O",
        "Group": "Multi-functional",
        "pKa1": 8.93,
        "Log S0": 0.12,
        "MolLogP": -0.98,
    },
}


# =========================================================
# HELPER: INTERACTIVE 3D VIEWER IN SIDEBAR (3Dmol.js)
# =========================================================
def render_3dmol_viewer(smiles_str, height=270):
    encoded_smiles = urllib.parse.quote(str(smiles_str))
    sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/SDF?record_type=3d"
    fallback_img = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_smiles}/PNG?record_type=3d&image_size=300x300"

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    </head>
    <body style="margin:0; padding:0; background-color:#ffffff;">
      <div id="viewer3d" style="width: 100%; height: {height}px; position: relative; border: 1px solid #cbd5e1; border-radius: 8px;"></div>
      <script>
        let container = document.getElementById('viewer3d');
        fetch('{sdf_url}')
          .then(response => {{
            if (!response.ok) throw new Error("HTTP " + response.status);
            return response.text();
          }})
          .then(data => {{
            let viewer = $3Dmol.createViewer(container, {{backgroundColor: 'white'}});
            viewer.addModel(data, "sdf");
            
            viewer.setStyle({{}}, {{
                stick: {{radius: 0.07, colorscheme: "Jmol"}},
                sphere: {{scale: 0.38, colorscheme: "Jmol"}}
            }});
            viewer.zoomTo();
            viewer.render();
          }})
          .catch(err => {{
            container.innerHTML = '<img src="{fallback_img}" style="width:100%; height:auto; border-radius:8px;" alt="3D Structure"/>';
          }});
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=height + 15)


def get_clean_dataframe(db_dict):
    df = pd.DataFrame(db_dict).T
    df.index.name = "Compound Name"
    cols = ["Smiles", "Group", "pKa1", "Log S0", "MolLogP"]
    return df[cols]


# ==========================================
# 4. ENGINE INFERENSI FORWARD CHAINING
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
    sol_log = data.get("Log S0", 0.0)

    S0 = 10**sol_log if sol_log is not None else 0.0
    smiles = data.get("Smiles", "")

    logs.append(f"⚙️ [INIT] Evaluasi Senyawa: {compound_name}")
    logs.append(
        f"   ↳ Group: {group} | pKa1: {pKa1} | MolLogP: {mollogp} | Intrinsic Log S0: {sol_log}"
    )

    if not is_pure_organic(smiles):
        logs.append(
            "❌ [FILTER REJECTED] Senyawa bukan molekul organik murni."
        )
        return (
            0.0,
            0.0,
            "Rejected",
            "#ef4444",
            logs,
            "Stot < 0.1 mol/L",
        )

    logs.append("✅ [FILTER PASSED] Molekul Organik Murni Terdeteksi.")

    # Executing Forward Chaining Rules
    if pKa1 is None:
        logs.append(
            f"🔹 [RULE JALUR LIPOFILISITAS FIRED] pKa1 = None → Evaluasi berbasis MolLogP ({mollogp})."
        )
        S_total = S0
    elif group == "Carboxylic Acid" or (
        group == "Multi-functional" and pKa1 < 7
    ):
        logs.append(
            f"🔹 [RULE ASAM LEMAH FIRED] IF Group == '{group}' DAN pKa1 ({pKa1}) DAN pH ({target_pH})"
        )
        exponent = target_pH - pKa1
        ion_factor = 10**exponent
        S_total = S0 * (1 + ion_factor)
        if target_pH > pKa1:
            logs.append(
                f"   ↳ pH ({target_pH}) > pKa1 ({pKa1}) → Form terionisasi (A-) mendominasi → Kelarutan Meningkat Signifikan."
            )
        else:
            logs.append(
                f"   ↳ pH ({target_pH}) <= pKa1 ({pKa1}) → Form netral HA mendominasi."
            )
    elif group == "Amine":
        logs.append(
            f"🔹 [RULE BASA LEMAH FIRED] IF Group == 'Amine' DAN pKa1 ({pKa1}) DAN pH ({target_pH})"
        )
        exponent = pKa1 - target_pH
        ion_factor = 10**exponent
        S_total = S0 * (1 + ion_factor)
        if target_pH < pKa1:
            logs.append(
                f"   ↳ pH ({target_pH}) < pKa1 ({pKa1}) → Terprotonasi (BH+) mendominasi → Kelarutan Meningkat."
            )
        else:
            logs.append(
                f"   ↳ pH ({target_pH}) >= pKa1 ({pKa1}) → Form netral B mendominasi."
            )
    elif group == "Phenol":
        logs.append(
            f"🔹 [RULE FENOL FIRED] IF Group == 'Phenol' DAN pKa1 ({pKa1}) DAN pH ({target_pH})"
        )
        exponent = target_pH - pKa1
        ion_factor = 10**exponent
        S_total = S0 * (1 + ion_factor)
    else:
        logs.append(
            f"🔹 [RULE NETRAL FIRED] IF Group == '{group}' → Tidak terionisasi signifikan pada pH normal."
        )
        S_total = S0

    # KLASIFIKASI KELARUTAN DISESUAIKAN PRESISI DENGAN TABEL STANDAR
    if S_total >= 100.0:
        cat, color = "Sangat Mudah Larut", "#16a34a"
        range_str = "Stot ≥ 100.0 mol/L"
    elif S_total >= 30.0:
        cat, color = "Mudah Larut", "#2563eb"
        range_str = "30.0 ≤ Stot < 100.0 mol/L"
    elif S_total >= 10.0:
        cat, color = "Larut", "#0284c7"
        range_str = "10.0 ≤ Stot < 30.0 mol/L"
    elif S_total >= 1.0:
        cat, color = "Agak Sukar Larut", "#d97706"
        range_str = "1.0 ≤ Stot < 10.0 mol/L"
    elif S_total >= 0.1:
        cat, color = "Sukar Larut", "#dc2626"
        range_str = "0.1 ≤ Stot < 1.0 mol/L"
    else:
        cat, color = "Praktis Tidak Larut", "#991b1b"
        range_str = "Stot < 0.1 mol/L"

    logs.append(
        f"🏁 [FINAL VERDICT] Total Kelarutan Stot = {S_total:.4f} mol/L → Kategori: {cat} ({range_str})"
    )
    return S0, S_total, cat, color, logs, range_str


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
# 6. SIDEBAR INPUT
# ==========================================
st.sidebar.header("⚙ Menu Navigasi & Input")

compound_options = ["-- Pilih Senyawa untuk Memulai --"] + list(
    AQSOL_IUPAC_DATABASE.keys()
)
selected_compound = st.sidebar.selectbox(
    "Pilih Senyawa Organik:", compound_options
)

# ==========================================
# 7. LOGIKA TAMPILAN PAGE
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
    st.dataframe(
        get_clean_dataframe(AQSOL_IUPAC_DATABASE), use_container_width=True
    )

else:
    compound_data = AQSOL_IUPAC_DATABASE[selected_compound]
    compound_data["name"] = selected_compound

    target_pH = st.sidebar.slider(
        "Atur pH Environment Pelarut:", 1.0, 14.0, 7.4, step=0.1
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📷 Visualisasi 3D")
    st.sidebar.caption(
        "💡 *Klik & geser untuk memutar atau memperbesar molekul*"
    )
    render_3dmol_viewer(compound_data.get("Smiles", ""), height=250)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Physical Descriptors")
    st.sidebar.text(f"Smiles: {compound_data.get('Smiles', 'N/A')}")
    st.sidebar.text(f"Group: {compound_data.get('Group', 'N/A')}")
    st.sidebar.text(f"MolLogP: {compound_data.get('MolLogP', 'N/A')}")
    st.sidebar.text(f"Log S0: {compound_data.get('Log S0', 'N/A')}")

    # Run Forward Chaining Engine
    S0_val, S_total, category, cat_color, rule_logs, range_str = (
        forward_chaining_engine(compound_data, target_pH)
    )

    # Display Metric Cards (Rentang S_tot Ditaruh Langsung Di Bawah Kategori)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">S0 (INTRINSIK)</div>
                <div class="metric-value">{S0_val:.6f} <span style="font-size:12px">mol/L</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">GUGUS FUNGSI</div>
                <div class="metric-value" style="color:#2563eb">{compound_data.get('Group', 'Unknown')}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">TINGKAT KELARUTAN</div>
                <div class="metric-value" style="color:{cat_color}; font-size:17px">{category}</div>
                <div class="metric-sub">Stot = {S_total:.4f} mol/L</div>
                <div class="metric-range">({range_str})</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col4:
        pka1_val = compound_data.get("pKa1")
        pka_display = f"{pka1_val}" if pka1_val is not None else "N/A"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">pKa1 (IUPAC)</div>
                <div class="metric-value" style="color:#d97706">{pka_display}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigasi Tab
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
            forward_chaining_engine(compound_data, ph)[1] for ph in pH_array
        ]

        fig = go.Figure()

        # Line Plot Utama
        fig.add_trace(
            go.Scatter(
                x=pH_array,
                y=S_array,
                mode="lines",
                name="Profil Kelarutan",
                line=dict(color="#2563eb", width=3),
            )
        )

        # State Target Saat Ini
        fig.add_trace(
            go.Scatter(
                x=[target_pH],
                y=[S_total],
                mode="markers+text",
                name=f"State Saat Ini (pH {target_pH})",
                marker=dict(color="#dc2626", size=12, symbol="diamond"),
                text=[f"  {S_total:.4f} mol/L"],
                textposition="top right",
                textfont=dict(color="#1e3a8a", size=13),
            )
        )

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
                title="<b>Total Kelarutan Stot (mol/L) - Skala Log</b>",
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
        st.dataframe(
            get_clean_dataframe(AQSOL_IUPAC_DATABASE), use_container_width=True
        )
