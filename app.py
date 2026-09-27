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

# ==========================================
# 3. DATABASE GABUNGAN AQSOLDB + IUPAC PKA
# ==========================================
AQSOL_IUPAC_DATABASE = {
    "Aspirin (Acetylsalicylic acid)": {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "type": "Weak Acid",
        "fg": ["carboxylic_acid", "ester"],
        "pKa1": 3.5,
        "pKa2": None,
        "S0": 3.3,
        "logP": 1.19,
        "tpsa": 63.6,
        "h_donors": 1,
        "h_acceptors": 4,
    },
    "Ibuprofen": {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "type": "Weak Acid",
        "fg": ["carboxylic_acid"],
        "pKa1": 4.4,
        "pKa2": None,
        "S0": 0.021,
        "logP": 3.5,
        "tpsa": 37.3,
        "h_donors": 1,
        "h_acceptors": 2,
    },
    "Phenol": {
        "smiles": "Oc1ccccc1",
        "type": "Phenol",
        "fg": ["phenol"],
        "pKa1": 9.95,
        "pKa2": None,
        "S0": 83.0,
        "logP": 1.46,
        "tpsa": 20.2,
        "h_donors": 1,
        "h_acceptors": 1,
    },
    "Paracetamol (Acetaminophen)": {
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "type": "Phenol",
        "fg": ["phenol", "amide"],
        "pKa1": 9.5,
        "pKa2": None,
        "S0": 14.0,
        "logP": 0.46,
        "tpsa": 49.3,
        "h_donors": 2,
        "h_acceptors": 2,
    },
    "Procaine": {
        "smiles": "CCN(CC)CCOC(=O)c1ccc(N)cc1",
        "type": "Weak Base",
        "fg": ["amine", "ester"],
        "pKa1": 8.9,
        "pKa2": None,
        "S0": 0.5,
        "logP": 2.14,
        "tpsa": 49.3,
        "h_donors": 1,
        "h_acceptors": 4,
    },
    "Benzenesulfonic Acid": {
        "smiles": "c1ccc(cc1)S(=O)(=O)O",
        "type": "Strong Acid",
        "fg": ["sulfonic_acid"],
        "pKa1": -2.8,
        "pKa2": None,
        "S0": 500.0,
        "logP": -0.18,
        "tpsa": 62.8,
        "h_donors": 1,
        "h_acceptors": 3,
    },
    "Ethyl Acetate": {
        "smiles": "CCOC(C)=O",
        "type": "Neutral",
        "fg": ["ester"],
        "pKa1": None,
        "pKa2": None,
        "S0": 83.0,
        "logP": 0.73,
        "tpsa": 26.3,
        "h_donors": 0,
        "h_acceptors": 2,
    },
    "Acetone": {
        "smiles": "CC(C)=O",
        "type": "Neutral",
        "fg": ["ketone"],
        "pKa1": None,
        "pKa2": None,
        "S0": 1000.0,
        "logP": -0.24,
        "tpsa": 17.1,
        "h_donors": 0,
        "h_acceptors": 1,
    },
    "Amoxicillin": {
        "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(c3ccc(O)cc3)N)C(=O)O)C",
        "type": "Amphoteric",
        "fg": ["amine", "carboxylic_acid", "phenol", "amide"],
        "pKa1": 2.4,
        "pKa2": 7.4,
        "S0": 3.4,
        "logP": 0.87,
        "tpsa": 158.0,
        "h_donors": 4,
        "h_acceptors": 7,
    },
}

# ==========================================
# 4. HELPER & INFERENCE ENGINE (5 LAPIS)
# ==========================================
def is_pure_organic(smiles):
    if "." in smiles:
        return False
    metals = ["Na", "K", "Ca", "Mg", "Fe", "Zn", "Li", "Ba"]
    for m in metals:
        if m in smiles:
            return False
    return True

def forward_chaining_engine(data, target_pH):
    logs = []
    smiles = data["smiles"]

    logs.append(f"⚙️ [INIT] Evaluasi Senyawa: {data.get('name', 'Target')}")
    logs.append(f"   ↳ Structure SMILES: {smiles}")

    if not is_pure_organic(smiles):
        logs.append("❌ [FILTER REJECTED] Senyawa bukan molekul organik murni.")
        return 0.0, "Rejected", "#ef4444", logs

    logs.append("✅ [FILTER PASSED] Molekul Organik Murni Terdeteksi.")

    c_type = data["type"]
    pKa1 = data["pKa1"]
    pKa2 = data["pKa2"]
    S0 = data["S0"]
    fg_list = data["fg"]

    # RULE C: Neutral / Non-Ionizable
    if c_type == "Neutral" or pKa1 is None:
        logs.append(f"🔹 [RULE C FIRED] IF FunctionalGroup ∈ {fg_list} → Non-Ionizable / Independent pH.")
        logs.append(f"   ↳ Kelarutan ditentukan oleh parameter fisik: LogP={data['logP']}, TPSA={data['tpsa']}.")
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
        pI = (pKa1 + pKa2) / 2
        logs.append(f"   ↳ Titik Isoelektrik (pI) = {pI:.2f}.")
        if abs(target_pH - pI) < 0.5:
            logs.append(f"   ↳ Pada pH {target_pH} (~pI): Dominan Zwitterion → Kelarutan Minimum.")
        elif target_pH < pKa1:
            logs.append(f"   ↳ Pada pH {target_pH} (< pKa1): Spesies Kationik mendominasi (+1).")
        else:
            logs.append(f"   ↳ Pada pH {target_pH} (> pKa2): Spesies Anionik mendominasi (-1).")
        S_total = S0 * (1 + 10 ** (target_pH - pKa2) + 10 ** (pKa1 - target_pH))

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
    st.sidebar.text(f"SMILES: {compound_data['smiles']}")
    st.sidebar.text(f"MolLogP: {compound_data['logP']}")
    st.sidebar.text(f"TPSA: {compound_data['tpsa']} Å²")
    st.sidebar.text(f"H-Donors: {compound_data['h_donors']}")
    st.sidebar.text(f"H-Acceptors: {compound_data['h_acceptors']}")

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
            <div class="metric-value" style="color:#2563eb">{compound_data['type']}</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-label">KLASIFIKASI USP</div>
            <div class="metric-value" style="color:{cat_color}; font-size:18px">{category}</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        pka_display = f"{compound_data['pKa1']}" if compound_data["pKa1"] else "N/A"
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
                textfont=dict(color="#1e3a8a", size=13)
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
                linecolor="#94a3b8"
            ),
            yaxis=dict(
                title="<b>Total Kelarutan (mg/mL) - Skala Log</b>",
                title_font=dict(color="#1e3a8a", size=14),
                tickfont=dict(color="#0f172a", size=12),
                gridcolor="#cbd5e1",
                linecolor="#94a3b8",
                type="log"
            ),
            legend=dict(
                font=dict(color="#0f172a", size=12),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#cbd5e1",
                borderwidth=1
            ),
            height=440,
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Data AqSolDB + IUPAC Joined Table")
        df_db = pd.DataFrame(AQSOL_IUPAC_DATABASE).T
        st.dataframe(df_db, use_container_width=True)