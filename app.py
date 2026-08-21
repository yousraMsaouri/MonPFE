import os
import io
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# CONFIG PAGE
# ==============================================================================
st.set_page_config(
    page_title="Jacob Delafon — Prévisions IA",
    page_icon="🛁",
    layout="wide",
)

# Dossier du script, pour que les chemins fonctionnent peu importe d'où on lance
# "streamlit run" (et peu importe la machine / le dossier de travail courant).
DOSSIER_SCRIPT = Path(__file__).parent

#--------------------------------------------------------------------------------------------------------------------------
# Définition globale des chemins pour éviter les erreurs de scope (NameError)
CHEMIN_MODELE = DOSSIER_SCRIPT / "models" / "super_blending_optuna_pca.joblib"
CHEMIN_METRICS = DOSSIER_SCRIPT / "models" / "metrics_stacking.json"

# Chemin vers le dossier SharePoint synchronisé via OneDrive
DOSSIER_USER = Path.home()

# Recherche automatique du dossier OneDrive (pour s'adapter à ton PC et à celui de ton responsable)
CHEMIN_DATA_SHAREPOINT = (
    DOSSIER_USER
    / "OneDrive - Kohler Co"
    / "Pricing & Marketing Intelligence-projets -satges - Project AIR"
    / "01_Export_Courant"
    / "prévisions_futures_6M_2027.csv"
)

# Secours : Si le fichier n'est pas trouvé dans OneDrive, on prend le fichier local dans /data
if CHEMIN_DATA_SHAREPOINT.exists():
    CHEMIN_DATA = CHEMIN_DATA_SHAREPOINT
else:
    CHEMIN_DATA = DOSSIER_SCRIPT / "data" / "prévisions_futures_6M_2027.csv"

# ==============================================================================
# CONFIGURATION DES SCÉNARIOS CONJONCTURELS AUTOMATIQUES (MATRICE EXPERT)
# ==============================================================================
SCENARIOS_MARCHE = {
    "Aucun ajustement (Normal)": {"impact": 0.0, "description": "Prévisions de base générées par l'IA."},
    "Campagne Marketing Majeure": {"impact": 0.15, "description": "Hausse automatique de +15% des volumes cibles."},
    "Signature Gros Contrat (Hôtellerie)": {"impact": 0.25, "description": "Forte hausse automatique de +25% sur le marché."},
    "Ralentissement Économique Pays": {"impact": -0.10, "description": "Baisse automatique de -10% due à la contraction de la demande."},
    "Pénurie de Matières Premières / Logistique": {"impact": -0.20, "description": "Restriction des volumes de -20% pour cause de rupture supply."}
}

# ==============================================================================
# CHARGEMENT DU MAPE DEPUIS LES MÉTRIQUES DU NOUVEAU MODÈLE
# ==============================================================================
def charger_mape(defaut=0.2654):
    if CHEMIN_METRICS.exists():
        try:
            with open(CHEMIN_METRICS, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return float(metrics.get("mape", defaut)), metrics
        except Exception:
            return defaut, None
    return defaut, None


MAPE, METRICS_MODELE = charger_mape()

# Le vrai MAPE peut être très élevé sur des petites quantités (division par de petites valeurs).
# On plafonne la zone AFFICHÉE pour rester lisible, sans jamais cacher le vrai chiffre (visible en sidebar).
MAPE_PLAFOND = 0.15
MAPE_AFFICHAGE = min(MAPE, MAPE_PLAFOND)

# Style CSS custom (cartes blanches + bannière SKU, repris du dashboard "Excel")
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.03); border: 1px solid #eaeaea; }
    .sku-banner { background-color: #ffffff; padding: 15px; border-radius: 8px;
                  border: 1px solid #e0e0e0; text-align: center; font-size: 1.15rem;
                  font-weight: 500; margin-bottom: 20px; color: #2c3e50; }
    .info-box { background-color: #ffffff; border: 1px solid #c7cbd1; border-radius: 8px;
                padding: 12px 16px; margin-bottom: 16px; color: #3a3f44;
                font-size: 0.85rem; line-height: 1.5; }
    h1 { color: #1a1a1a; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Tableau de Bord Prédictif — Supply Chain Jacob Delafon")
st.markdown("---")

# ==============================================================================
# CORRESPONDANCE DES MOIS (robuste : accents, majuscules, anglais, numérique)
# ==============================================================================
CORRECTIONS_ENCODAGE = {
    "AoÃ»t": "Août",
    "FÃ©vrier": "Février",
    "DÃ©cembre": "Décembre",
    "Ao?t": "Août",
    "F?vrier": "Février",
    "D?cembre": "Décembre",
}

MOIS_FR = {
    "Janvier": 1, "Février": 2, "Mars": 3, "Avril": 4, "Mai": 5, "Juin": 6,
    "Juillet": 7, "Août": 8, "Septembre": 9, "Octobre": 10, "Novembre": 11, "Décembre": 12,
}

MOIS_EN = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Sept": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def normaliser(texte):
    """Minuscule + suppression des accents, pour matcher peu importe l'encodage/la casse."""
    import unicodedata
    txt = unicodedata.normalize("NFKD", str(texte).strip())
    return "".join(c for c in txt if not unicodedata.combining(c)).lower()


# Table de correspondance normalisée (fusion FR + EN)
MOIS_NORMALISES = {normaliser(k): v for k, v in {**MOIS_FR, **MOIS_EN}.items()}


def convertir_mois(valeur):
    """Retourne le numéro du mois (1-12) ou None si non reconnu (au lieu de forcer Janvier)."""
    txt_norm = normaliser(valeur)
    if txt_norm in MOIS_NORMALISES:
        return MOIS_NORMALISES[txt_norm]
    # Cas où le mois est déjà numérique ("1", "01", "1.0"...)
    try:
        num = int(float(txt_norm))
        if 1 <= num <= 12:
            return num
    except ValueError:
        pass
    return None


# ==============================================================================
# CHARGEMENT DU NOUVEAU MODÈLE + DU CSV DE PRÉVISIONS (mis en cache)
# ==============================================================================
@st.cache_resource
def charger_elements():
    if not CHEMIN_MODELE.exists():
        st.error(f"❌ Impossible de trouver le modèle IA dans : {CHEMIN_MODELE}")
        return None, None
    if not CHEMIN_DATA.exists():
        st.error(f"❌ Impossible de trouver le fichier dans : {CHEMIN_DATA}")
        return None, None

    model = joblib.load(CHEMIN_MODELE)

    # Détection automatique du séparateur (',' ',' ou ';' selon la machine/export Excel)
    df = None
    dernier_erreur = None
    for encodage in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            df = pd.read_csv(
                CHEMIN_DATA,
                encoding=encodage,
                sep=None,          # auto-détection du séparateur
                engine="python",   # requis pour sep=None
            )
            break
        except Exception as e:
            dernier_erreur = e
            continue

    if df is None:
        st.error(f"❌ Impossible de lire le fichier CSV (encodage/séparateur non reconnu) : {dernier_erreur}")
        return None, None

    df.columns = df.columns.str.strip()
    return model, df


model_ia, df_clean = charger_elements()

if model_ia is None or df_clean is None:
    st.stop()

# ==============================================================================
# DÉTECTION AUTOMATIQUE DES COLONNES
# ==============================================================================
col_pays = None
col_canal = None
col_cat = None
col_year = None
col_month = None
col_sku = None
col_designation = None
col_realiste = None
col_pessimiste = None
col_optimiste = None
col_historique = None

for c in df_clean.columns:
    c_low = c.lower()
    if "country" in c_low or "pays" in c_low:
        col_pays = c
    elif "channel" in c_low or "canal" in c_low:
        col_canal = c
    elif "cat" in c_low and "sous" not in c_low:
        col_cat = c
    elif "year" in c_low or "année" in c_low or "annee" in c_low:
        col_year = c
    elif ("month" in c_low or "mois" in c_low) and "num" not in c_low:
        col_month = c
    elif "material code name" in c_low or "designation" in c_low or "désignation" in c_low:
        col_designation = c
    elif "material code" in c_low or c_low == "sku" or "code article" in c_low:
        col_sku = c
    elif "name" in c_low and col_designation is None:
        col_designation = c

for c in df_clean.columns:
    c_low = c.lower()
    if "realiste" in c_low or "réaliste" in c_low or "prediction" in c_low:
        col_realiste = c
    elif "pessimiste" in c_low:
        col_pessimiste = c
    elif "optimiste" in c_low:
        col_optimiste = c
    elif "historique" in c_low or "vrai volume" in c_low or "supply" in c_low:
        col_historique = c

if not col_cat:
    for c in df_clean.columns:
        if "cat" in c.lower():
            col_cat = c

colonnes_manquantes = [
    nom for nom, val in [
        ("Réaliste/Prédiction", col_realiste),
        ("Année", col_year),
        ("Mois", col_month),
    ] if val is None
]

if colonnes_manquantes:
    st.error(f"❌ Erreur de structure : colonnes introuvables → {', '.join(colonnes_manquantes)}")
    st.stop()

# ==============================================================================
# NETTOYAGE DES MOIS (encodage + mapping numérique) + DATE
# ==============================================================================
df_clean[col_month] = df_clean[col_month].astype(str).str.strip()
df_clean[col_month] = df_clean[col_month].replace(CORRECTIONS_ENCODAGE)
df_clean["_Num_Mois"] = df_clean[col_month].apply(convertir_mois)

nb_mois_non_reconnus = df_clean["_Num_Mois"].isna().sum()
if nb_mois_non_reconnus > 0:
    exemples = df_clean.loc[df_clean["_Num_Mois"].isna(), col_month].unique()[:5]
    st.sidebar.warning(
        f"{nb_mois_non_reconnus} ligne(s) avec un mois non reconnu ont été ignorées.\n"
        f"Exemples : {', '.join(map(str, exemples))}"
    )

df_clean = df_clean.dropna(subset=["_Num_Mois"]).copy()
df_clean["_Num_Mois"] = df_clean["_Num_Mois"].astype(int)
df_clean["Month_Date"] = pd.to_datetime(
    df_clean[col_year].astype(str) + "-" + df_clean["_Num_Mois"].astype(str) + "-01",
    errors="coerce",
)

for col in [col_pays, col_canal, col_cat]:
    if col:
        df_clean[col] = df_clean[col].astype(str).str.strip()

# ==============================================================================
# SIDEBAR — LOGO + FILTRES
# ==============================================================================
chemin_logo = DOSSIER_SCRIPT / "assets" / "jacob-delafon-logo.png"
if chemin_logo.exists():
    st.sidebar.image(str(chemin_logo), width=200)
else:
    st.sidebar.caption(
        f"ℹ️ Logo non trouvé. Copie ton fichier .png dans :\n`{(DOSSIER_SCRIPT / 'assets').resolve()}`\n"
        f"sous le nom `jacob-delafon-logo.png`."
    )

st.sidebar.header("Filtres Globaux")

lignes_info = []
if METRICS_MODELE:
    lignes_info.append(
        f"• Modèle : {CHEMIN_MODELE.name}<br>"
        f"• MAPE réel : {METRICS_MODELE.get('mape', MAPE)*100:.2f}% | "
        f"R² : {METRICS_MODELE.get('r2', 0):.4f} | "
        f"MAE : {METRICS_MODELE.get('mae', 0):.2f} pcs"
    )
else:
    lignes_info.append(f"• Modèle chargé : {CHEMIN_MODELE.name}<br>• MAPE par défaut : {MAPE*100:.2f}%")

if MAPE > MAPE_PLAFOND:
    lignes_info.append(
        f"• Zone graphique plafonnée à ±{MAPE_PLAFOND*100:.0f}% pour rester visuellement stable "
        f"(le MAPE réel calculé est de {MAPE*100:.2f}%)."
    )

st.sidebar.markdown(
    f'<div class="info-box">{"<br><br>".join(lignes_info)}</div>',
    unsafe_allow_html=True,
)

if not METRICS_MODELE:
    with st.sidebar.expander("Pourquoi la valeur par défaut ?"):
        st.write(f"Fichier recherché mais introuvable :\n``{CHEMIN_METRICS.resolve()}``")
        st.write("Vérifie que le nouveau fichier métriques existe sous le nom exact : `metrics_stacking.json`.")

# --- Pays ---
liste_pays = ["Tous"] + sorted(df_clean[col_pays].dropna().unique().tolist()) if col_pays else ["Tous"]
pays_selectionne = st.sidebar.selectbox("Pays", liste_pays)

df_apres_pays = df_clean.copy()
if pays_selectionne != "Tous" and col_pays:
    df_apres_pays = df_apres_pays[df_apres_pays[col_pays] == pays_selectionne]

# --- Canal ---
liste_canaux = ["Tous"] + sorted(df_apres_pays[col_canal].dropna().unique().tolist()) if col_canal else ["Tous"]
canal_selectionne = st.sidebar.selectbox("Canal", liste_canaux)

df_apres_canal = df_apres_pays.copy()
if canal_selectionne != "Tous" and col_canal:
    df_apres_canal = df_apres_canal[df_apres_canal[col_canal] == canal_selectionne]

# --- Catégorie ---
liste_categories = ["Tous"] + sorted(df_apres_canal[col_cat].dropna().unique().tolist()) if col_cat else ["Tous"]
cat_selectionnee = st.sidebar.selectbox("Catégorie", liste_categories)

df_filtre = df_apres_canal.copy()
if cat_selectionnee != "Tous" and col_cat:
    df_filtre = df_filtre[df_filtre[col_cat] == cat_selectionnee]

# ------------------------------------------------------------------------------
# 🛠️ AJOUT EXCLUSIF : SYSTEME EXPERT AVEC FENÊTRE POP-UP (st.dialog)
# ------------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Ajustement Marché & Conjoncture")

# Initialisation des variables dans la session Streamlit pour persister les états
if "scenario_choisi" not in st.session_state:
    st.session_state.scenario_choisi = "Aucun ajustement (Normal)"
if "facteur_impact" not in st.session_state:
    st.session_state.facteur_impact = 1.0

# Définition de la fonction de la fenêtre Pop-up
@st.dialog("⚙️ Configurer l'Ajustement Conjoncturel")
def ouvrir_popup_ajustement(pays):
    st.markdown(f"**Pays ciblé :** `{pays}`")
    st.write("Sélectionnez l'événement macro-économique ou commercial qui impacte ce marché. L'application appliquera automatiquement le coefficient requis.")
    
    # Liste déroulante des scénarios
    choix = st.selectbox(
        "Type d'ajustement marché :", 
        list(SCENARIOS_MARCHE.keys()),
        index=list(SCENARIOS_MARCHE.keys()).index(st.session_state.scenario_choisi)
    )
    
    # Affichage de la description de l'impact
    st.caption(f"ℹ️ *Action du modèle : {SCENARIOS_MARCHE[choix]['description']}*")
    
    if st.button("Appliquer l'ajustement", use_container_width=True):
        st.session_state.scenario_choisi = choix
        st.session_state.facteur_impact = 1.0 + SCENARIOS_MARCHE[choix]["impact"]
        st.rerun()

# Bouton dans la sidebar pour déclencher l'ouverture de la fenêtre Pop-up
if st.sidebar.button("⚡ Ouvrir les scénarios marché", use_container_width=True):
    ouvrir_popup_ajustement(pays_selectionne)

# Affichage du statut actuel de l'ajustement dans la sidebar
st.sidebar.info(f"**Statut actuel :**\n{st.session_state.scenario_choisi}")

# Application automatique directe du coefficient sur les volumes filtrés
df_filtre[col_realiste] = df_filtre[col_realiste] * st.session_state.facteur_impact

# --- Suite des filtres classiques ---
vue = st.sidebar.radio("Vue", ["Vue Globale", "Vue par SKU"])

mois_dispo = sorted(df_filtre["Month_Date"].dropna().unique())
nb_mois_max = max(len(mois_dispo), 1)
nb_mois = st.sidebar.slider("Mois à prédire", 1, nb_mois_max, nb_mois_max)
if len(mois_dispo) > nb_mois:
    mois_limites = mois_dispo[:nb_mois]
    df_filtre = df_filtre[df_filtre["Month_Date"].isin(mois_limites)]

sku_choisi = None
if vue == "Vue par SKU":
    if col_sku:
        skus_disponibles = sorted(df_filtre[col_sku].dropna().unique().tolist())
        if skus_disponibles:
            sku_choisi = st.sidebar.selectbox("SKU", skus_disponibles)
        else:
            st.sidebar.warning("Aucun SKU disponible pour cette sélection.")
    else:
        st.sidebar.warning("Aucune colonne SKU détectée dans le fichier.")

st.markdown(f"**Pays :** {pays_selectionne} | **Canal :** {canal_selectionne} | **Catégorie :** {cat_selectionnee}")
if st.session_state.scenario_choisi != "Aucun ajustement (Normal)":
    st.warning(f"🔄 **Ajustement Expert Actif sur '{pays_selectionne}'** : Le scénario *'{st.session_state.scenario_choisi}'* a été appliqué automatiquement aux volumes de prévision.")
st.markdown("---")


# ==============================================================================
# FONCTION UTILITAIRE : GRAPHIQUE BARRES + ZONE MAPE (CALIBRÉ)
# ==============================================================================
def tracer_graphique(df_pred, titre=None):
    fig, ax = plt.subplots(figsize=(14, 5.5))

    if not df_pred.empty:
        df_pred = df_pred.sort_values("Month_Date")
        
        # 1. Tracé des barres réalistes
        ax.bar(
            df_pred["Month_Date"], df_pred[col_realiste],
            width=18, color="#2ecc71", alpha=0.8, label="Prédiction réaliste (IA Blending)",
        )
        
        # 2. Calcul proportionnel du tunnel de confiance à ±15% après sommation
        pess = df_pred[col_realiste] * (1 - MAPE_AFFICHAGE)
        opt = df_pred[col_realiste] * (1 + MAPE_AFFICHAGE)
        
        # 3. Tracé de la zone rouge d'incertitude
        ax.fill_between(
            df_pred["Month_Date"], pess, opt,
            color="#e74c3c", alpha=0.12, edgecolor="#e74c3c", linestyle="--", linewidth=1,
            label=f"Zone Pess/Opt (±{MAPE_AFFICHAGE*100:.1f}%)",
        )

    if titre:
        ax.set_title(titre, loc="left", fontweight="bold")
    ax.set_ylabel("Quantité (unités)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=35, ha="right")
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


def export_excel(df_export, sheet_name="Prévisions"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer


# ==============================================================================
# VUE GLOBALE
# ==============================================================================
if vue == "Vue Globale":
    st.subheader(f"Vue Globale — {cat_selectionnee} / {canal_selectionne} / {pays_selectionne}")

    if df_filtre.empty:
        st.warning("Aucune donnée disponible pour cette combinaison de filtres.")
    else:
        df_macro = df_filtre.groupby("Month_Date")[col_realiste].sum().reset_index()

        col1, col2, col3 = st.columns(3)
        with col1:
            nb_skus = df_filtre[col_sku].nunique() if col_sku else "—"
            st.metric("SKUs analysés", f"{nb_skus:,}".replace(",", " ") if isinstance(nb_skus, int) else nb_skus)
        with col2:
            val_pred = df_macro[col_realiste].mean() if not df_macro.empty else 0
            st.metric("Prédiction moy.", f"{int(val_pred):,}".replace(",", " ") + " pcs")
        with col3:
            st.metric("Mois prédits", f"{len(df_macro)}")

        fig = tracer_graphique(df_macro)
        st.pyplot(fig)

        df_table = df_macro.copy().sort_values("Month_Date")
        df_table["Mois"] = df_table["Month_Date"].dt.strftime("%b %Y")
        
        df_table["Pessimiste"] = (df_table[col_realiste] * (1 - MAPE_AFFICHAGE)).round(0)
        df_table["Réaliste"] = df_table[col_realiste].round(0)
        df_table["Optimiste"] = (df_table[col_realiste] * (1 + MAPE_AFFICHAGE)).round(0)

        df_export = pd.DataFrame({
            "Catégorie": cat_selectionnee,
            "Canal": canal_selectionne,
            "Pays": pays_selectionne,
            "Période": df_table["Mois"],
            "Scénario Pessimiste (IA)": df_table["Pessimiste"],
            "Prédiction Réaliste (IA)": df_table["Réaliste"],
            "Scénario Optimiste (IA)": df_table["Optimiste"],
        })
        buffer_global = export_excel(df_export, "Synthèse Globale")

        st.markdown("---")
        col_titre_g, col_btn_g = st.columns([7, 3])
        with col_titre_g:
            st.markdown(f"### Synthèse globale : {cat_selectionnee} / {canal_selectionne} / {pays_selectionne}")
        with col_btn_g:
            st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="Télécharger l'export général (Excel)",
                data=buffer_global,
                file_name=f"Export_General_{cat_selectionnee}_{canal_selectionne}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download-excel-global",
                use_container_width=True,
            )

        st.dataframe(
            df_table[["Mois", "Pessimiste", "Réaliste", "Optimiste"]].set_index("Mois").style.format(lambda x: f"{x:,.0f}".replace(",", " ")),
            width="stretch",
        )

# ==============================================================================
# VUE PAR SKU
# ==============================================================================
elif vue == "Vue par SKU" and sku_choisi is not None:
    df_sku = df_filtre[df_filtre[col_sku] == sku_choisi].sort_values("Month_Date")

    if df_sku.empty:
        st.warning("Aucune donnée temporelle pour ce SKU spécifique.")
    else:
        nom_produit = (
            df_sku[col_designation].iloc[0] if col_designation and not df_sku[col_designation].isna().all()
            else "Désignation non renseignée"
        )

        st.markdown(
            f'<div class="sku-banner">Focus Produit — SKU: {sku_choisi} | {nom_produit}</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            sku_reduit = f"{str(sku_choisi)[:6]}..." if len(str(sku_choisi)) > 8 else sku_choisi
            st.metric("SKU Analysé", sku_reduit)
        with col2:
            val_f_sku = df_sku[col_realiste].mean()
            st.metric("Prédiction moy.", f"{int(val_f_sku):,}".replace(",", " ") + " pcs")
        with col3:
            st.metric("Mois prédits", f"{len(df_sku)}")

        fig = tracer_graphique(df_sku, titre="Prévisions Évolutives par SKU")
        st.pyplot(fig)

        df_table = df_sku.copy().sort_values("Month_Date")
        df_table["Mois"] = df_table["Month_Date"].dt.strftime("%b %Y")
        
        df_table["Pessimiste"] = (df_table[col_realiste] * (1 - MAPE_AFFICHAGE)).round(0)
        df_table["Réaliste"] = df_table[col_realiste].round(0)
        df_table["Optimiste"] = (df_table[col_realiste] * (1 + MAPE_AFFICHAGE)).round(0)

        colonnes_export = {}
        if col_year: colonnes_export[col_year] = "Année"
        if col_month: colonnes_export[col_month] = "Mois Nom"
        colonnes_export["Mois"] = "Période"
        if col_canal: colonnes_export[col_canal] = "Canal"
        if col_pays: colonnes_export[col_pays] = "Pays"
        if col_cat: colonnes_export[col_cat] = "Catégorie"
        if col_sku: colonnes_export[col_sku] = "Code Article (SKU)"
        if col_designation: colonnes_export[col_designation] = "Désignation Article"
        colonnes_export["Pessimiste"] = "Scénario Pessimiste (IA)"
        colonnes_export["Réaliste"] = "Prédiction Réaliste (IA)"
        colonnes_export["Optimiste"] = "Scénario Optimiste (IA)"

        cols_existantes = [c for c in colonnes_export.keys() if c in df_table.columns]
        df_export_complet = df_table[cols_existantes].rename(columns=colonnes_export)
        buffer = export_excel(df_export_complet, "Prédictions IA")

        col_titre, col_btn = st.columns([7, 3])
        with col_titre:
            st.markdown("### Tableau des prédictions")
        with col_btn:
            st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="Télécharger l'export complet (Excel)",
                data=buffer,
                file_name=f"Predictions_Completes_SKU_{sku_choisi}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download-excel-sku",
                use_container_width=True,
            )

        st.dataframe(
            df_table[["Mois", "Pessimiste", "Réaliste", "Optimiste"]].set_index("Mois").style.format(lambda x: f"{x:,.0f}".replace(",", " ")),
            width="stretch",
        )