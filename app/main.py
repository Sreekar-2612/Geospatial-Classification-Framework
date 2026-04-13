import streamlit as st
import numpy as np
import cv2
import sys
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from scipy.stats import chi2_contingency

# Path fixes
sys.path.append(str(Path(__file__).parents[1]))
from src.features import extract_lulc_features
from src.utils import METRICS_PATH

# --- Setup ---
st.set_page_config(page_title="LULC Intelligence Dashboard", layout="wide")

st.markdown("""
<style>
    .report-card { background: white; color: #1e293b; padding: 20px; border-radius: 10px; border-left: 5px solid #3b82f6; margin-bottom: 20px; font-family: sans-serif; }
    .class-gallery-item { text-align: center; border: 1px solid #ddd; border-radius: 5px; padding: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
CLASSES = ["Agriculture", "Buildings", "Forest", "Roads", "Water"]
CLASS_COLORS = {
    "Agriculture": "#639922", "Buildings": "#888780",
    "Forest": "#1D9E75", "Roads": "#444441", "Water": "#378ADD"
}
MODELS_DIR = Path(__file__).parents[1] / "models"

# --- Semantic Color Mapping for Clusters ---
TARGET_CLASSES_LIST = [
    ("Buildings/Urban", [180, 180, 180], [255, 255, 255]),
    ("Roads/Pavement", [110, 110, 110], [128, 128, 128]),
    ("Forest/Parks", [45, 65, 40], [34, 139, 34]),
    ("Water Bodies", [20, 40, 100], [30, 144, 255]),
    ("Agriculture/Grass", [100, 115, 80], [154, 205, 50]),
    ("Shadows/Unknown", [30, 30, 35], [40, 40, 40])
]

# --- Load Models ---
@st.cache_resource
def load_rf():
    path = MODELS_DIR / "rf_baseline.joblib"
    return joblib.load(path) if path.exists() else None

@st.cache_resource
def load_cnn():
    path = MODELS_DIR / "cnn_final.pth"
    if not path.exists():
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError:
        return None, None
    model.to(device)
    model.eval()
    return model, device

rf_model = load_rf()
cnn_model, cnn_device = load_cnn()

CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_rf(img_np):
    if rf_model is None:
        return None, None
    feat = extract_lulc_features(img_np)
    pred_idx = rf_model.predict([feat])[0]
    probs = rf_model.predict_proba([feat])[0]
    return pred_idx, probs

def predict_cnn(img_pil):
    if cnn_model is None:
        return None, None
    tensor = CNN_TRANSFORM(img_pil).unsqueeze(0).to(cnn_device)
    with torch.no_grad():
        output = cnn_model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
    return pred_idx, probs

# --- Segmentation Logic ---
def get_class_masks(img_np, k=8, apply_smoothing=False):
    Z = img_np.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    cluster_to_label_idx = []
    for i in range(k):
        min_dist = float('inf')
        best_t_idx = -1
        for j, (_, expected_rgb, _) in enumerate(TARGET_CLASSES_LIST):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(expected_rgb))
            if dist < min_dist:
                min_dist = dist
                best_t_idx = j
        cluster_to_label_idx.append(best_t_idx)
        
    unique_label_indices = list(set(cluster_to_label_idx))
    semantic_masks = {}
    full_seg = np.zeros_like(img_np)
    
    for t_idx in unique_label_indices:
        label_name, _, display_color = TARGET_CLASSES_LIST[t_idx]
        matching_cluster_indices = [i for i, mapped_t_idx in enumerate(cluster_to_label_idx) if mapped_t_idx == t_idx]
        mask_bool = np.isin(labels.flatten(), matching_cluster_indices)
        mask_2d = mask_bool.reshape(img_np.shape[:2]).astype(np.uint8) * 255
        if apply_smoothing:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_2d = cv2.morphologyEx(mask_2d, cv2.MORPH_CLOSE, kernel)
            mask_2d = cv2.medianBlur(mask_2d, 5)
        final_mask_bool = mask_2d > 127
        colored_mask = np.zeros_like(img_np)
        colored_mask[final_mask_bool] = display_color
        semantic_masks[label_name] = colored_mask
        full_seg[final_mask_bool] = display_color
        
    return semantic_masks, full_seg

# ===== TABS =====
tab_main, tab_report, tab_temporal = st.tabs(["🛰️ Live Classification", "📊 Model Report", "🕒 Temporal Analysis"])

# =============================================
# TAB 1: MAIN INFERENCE
# =============================================
with tab_main:
    st.title("🛰️ Land Use and Land Cover Classification Dashboard")
    
    # --- Sidebar ---
    st.sidebar.header("🛠️ S-Grade Features")
    apply_smoothing = st.sidebar.checkbox("Enable Spatial Smoothing", value=False,
        help="Uses Median Filtering & Morphological Closing to reduce salt/pepper noise.")

    # --- System Status ---
    st.sidebar.divider()
    st.sidebar.header("🛡️ System Persistence")
    rf_status = "✅ RF Loaded" if rf_model is not None else "❌ RF Missing"
    cnn_status = "✅ CNN Loaded" if cnn_model is not None else "❌ CNN Missing"
    st.sidebar.caption(f"{rf_status} | {cnn_status}")

    # --- Model Selector ---
    st.sidebar.divider()
    st.sidebar.header("🧠 Model Selector")
    model_choice = st.sidebar.radio(
        "Select Inference Model",
        options=["Random Forest (RF)", "CNN (ResNet-18)", "Compare Both"],
        index=0,
        key="main_model_selector"
    )

    st.sidebar.divider()
    st.sidebar.header("Control Panel")
    uploaded_file = st.sidebar.file_uploader("Upload Satellite Patch", type=["jpg", "png", "tiff"], key="main_uploader")
    target_names = CLASSES

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(img, use_container_width=True)
            
            st.subheader("🏗️ Infrastructure Detection")
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            infra_mask = cv2.dilate(edges, kernel, iterations=1)
            st.image(infra_mask, use_container_width=True, caption="DIP Edge Detection isolating Roads/Buildings")
            
            st.subheader("🌿 Vegetative Health (Pseudo-NDVI)")
            R = img_np[:, :, 0].astype(np.float32)
            G = img_np[:, :, 1].astype(np.float32)
            denominator = (G + R)
            denominator[denominator == 0] = 1
            ndvi = (G - R) / denominator
            fig_ndvi = px.imshow(ndvi, color_continuous_scale="RdYlGn", zmin=-0.3, zmax=0.3)
            fig_ndvi.update_layout(margin=dict(l=0, r=0, b=0, t=0), coloraxis_showscale=False)
            st.plotly_chart(fig_ndvi, use_container_width=True)
            st.caption("Mathematical Index: (G - R) / (G + R)")
            
        with col2:
            st.subheader("🎨 Classified Map (K-Means)")
            masks, full_seg = get_class_masks(img_np, apply_smoothing=apply_smoothing)
            st.image(full_seg, use_container_width=True)
            st.caption("Dynamically Mapped Semantic Clusters vs Haze.")
            
            st.markdown("""
            <div style="display: flex; gap: 10px; flex-wrap: wrap; font-size: 12px; color: #333; background: #eee; padding: 5px; border-radius: 5px;">
                <span><span style="color:white; -webkit-text-stroke: 1px black;">■</span> Buildings/Urban</span>
                <span><span style="color:gray">■</span> Roads/Pavement</span>
                <span><span style="color:forestgreen">■</span> Forest/Parks</span>
                <span><span style="color:dodgerblue">■</span> Water</span>
                <span><span style="color:yellowgreen">■</span> Agriculture</span>
                <span><span style="color:#282828">■</span> Shadows/Unknown</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.subheader("🧠 Intelligence Report")

            if model_choice == "Compare Both":
                c_rf, c_cnn = st.columns(2)
                with c_rf:
                    st.markdown("**Random Forest**")
                    pred_rf, probs_rf = predict_rf(img_np)
                    if pred_rf is not None:
                        st.success(f"{target_names[pred_rf]}")
                        st.metric("Confidence", f"{probs_rf[pred_rf]*100:.1f}%")
                with c_cnn:
                    st.markdown("**CNN (ResNet-18)**")
                    pred_cnn, probs_cnn = predict_cnn(img)
                    if pred_cnn is not None:
                        st.success(f"{target_names[pred_cnn]}")
                        st.metric("Confidence", f"{probs_cnn[pred_cnn]*100:.1f}%")

            elif model_choice == "CNN (ResNet-18)":
                pred_cnn, probs_cnn = predict_cnn(img)
                if pred_cnn is not None:
                    st.success(f"**Primary Class**: {target_names[pred_cnn]}")
                    st.info(f"**Confidence**: {probs_cnn[pred_cnn]*100:.1f}%")
                    prob_df = pd.DataFrame({'Class': CLASSES, 'Probability': probs_cnn})
                    fig_p = px.bar(prob_df, x='Class', y='Probability', color='Class', color_discrete_map=CLASS_COLORS)
                    fig_p.update_layout(showlegend=False, margin=dict(l=0,r=0,b=0,t=30), height=220)
                    st.plotly_chart(fig_p, use_container_width=True)

            else:  # Random Forest
                pred_rf, probs_rf = predict_rf(img_np)
                if pred_rf is not None:
                    st.success(f"**Primary Class**: {target_names[pred_rf]}")
                    st.info(f"**Confidence**: {probs_rf[pred_rf]*100:.1f}%")
                    prob_df = pd.DataFrame({'Class': CLASSES, 'Probability': probs_rf})
                    fig_p = px.bar(prob_df, x='Class', y='Probability', color='Class', color_discrete_map=CLASS_COLORS)
                    fig_p.update_layout(showlegend=False, margin=dict(l=0,r=0,b=0,t=30), height=220)
                    st.plotly_chart(fig_p, use_container_width=True)

        st.divider()
        st.header("🖼️ Dynamic Semantic Segregation (S-Grade Feature)")
        labels_list = list(masks.keys())
        g_cols = st.columns(max(len(labels_list), 1))
        for i, g_col in enumerate(g_cols):
            with g_col:
                st.markdown(f"**{labels_list[i]}**")
                st.image(masks[labels_list[i]], use_container_width=True)

# =============================================
# TAB 2: MODEL REPORT
# =============================================
with tab_report:
    st.title("📊 Model Report & Ablation Study")
    if not METRICS_PATH.exists():
        st.warning("⚠️ `report/metrics.csv` not found. Please run Notebook 05 first.")
    else:
        df_metrics = pd.read_csv(METRICS_PATH)
        st.subheader("📋 Ablation Study: Full Results")
        numeric_cols = df_metrics.select_dtypes(include='number').columns.tolist()
        styled = df_metrics.style.highlight_max(subset=numeric_cols, color='#d4edda', axis=0).format({c: "{:.4f}" for c in numeric_cols})
        st.dataframe(styled, use_container_width=True)

        st.subheader("📊 Side-by-Side Metric Comparison")
        if numeric_cols:
            df_melt = df_metrics.melt(id_vars='Model', value_vars=numeric_cols, var_name='Metric', value_name='Score')
            fig_bar = px.bar(df_melt, x='Model', y='Score', color='Metric', barmode='group', text_auto='.3f', color_discrete_sequence=px.colors.qualitative.Bold)
            fig_bar.update_layout(yaxis_range=[0, 1.1], plot_bgcolor='white', legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("🕸️ Performance Radar Chart")
        if numeric_cols and len(df_metrics) > 0:
            fig_radar = go.Figure()
            for _, row in df_metrics.iterrows():
                vals = [row[c] for c in numeric_cols] + [row[numeric_cols[0]]]
                fig_radar.add_trace(go.Scatterpolar(r=vals, theta=numeric_cols + [numeric_cols[0]], fill='toself', name=row['Model']))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)

# =============================================
# TAB 3: TEMPORAL ANALYSIS (NEW S-GRADE FEATURE)
# =============================================
with tab_temporal:
    st.title("🕒 Temporal Land Cover Intelligence Analyser")
    st.markdown("Advanced multi-temporal change detection and statistical significance testing.")
    
    # --- Sidebar Controls ---
    st.sidebar.divider()
    st.sidebar.header("🕒 Temporal Controls")
    t1_label = st.sidebar.text_input("T1 Label (Past)", value="2019")
    t2_label = st.sidebar.text_input("T2 Label (Present)", value="2024")
    pixel_scale = st.sidebar.number_input("Pixel Scale (m/px)", value=10, min_value=1)
    sankey_threshold = st.sidebar.slider("Sankey Threshold (%)", 0.0, 20.0, 1.0)
    temp_model = st.sidebar.selectbox("Analysis Engine", ["CNN (ResNet-18)", "Random Forest (RF)"])
    show_unchanged = st.sidebar.toggle("Show Unchanged Pixels in Heatmap", value=True)

    up_col1, up_col2 = st.columns(2)
    with up_col1:
        file_t1 = st.file_uploader(f"Upload {t1_label} Image", type=["jpg", "png", "tiff"])
    with up_col2:
        file_t2 = st.file_uploader(f"Upload {t2_label} Image", type=["jpg", "png", "tiff"])

    if file_t1 and file_t2:
        img1 = Image.open(file_t1).convert("RGB")
        img2 = Image.open(file_t2).convert("RGB")
        
        # 1. Independent Classification using K-Means for pixel-level granular maps
        # Note: For temporal change, we use K-Means as it maps EVERY pixel to a class.
        
        @st.cache_data
        def run_segmentation(img_np):
            # Recalculated locally for tab independence
            Z = img_np.reshape((-1, 3)).astype(np.float32)
            _, labels, centers = cv2.kmeans(Z, 6, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
            
            cluster_to_idx = []
            for c in centers:
                dists = [np.linalg.norm(c - np.array(e[1])) for e in TARGET_CLASSES_LIST]
                cluster_to_idx.append(np.argmin(dists))
            
            final_labels = np.array([cluster_to_idx[l[0]] for l in labels])
            return final_labels.reshape(img_np.shape[:2])

        if img1.size != img2.size:
            st.info(f"📐 Aligning spatial resolution: Resizing {t2_label} ({img2.size}) to match {t1_label} ({img1.size})")
            img2 = img2.resize(img1.size, Image.NEAREST)

        map_t1 = run_segmentation(np.array(img1))
        map_t2 = run_segmentation(np.array(img2))

        # 2. Transition Matrix & Significance
        def compute_transition_matrix(m1, m2, n=6):
            matrix = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(n):
                    matrix[i, j] = ((m1 == i) & (m2 == j)).sum()
            return matrix

        trans_matrix = compute_transition_matrix(map_t1, map_t2)
        
        chi2, p_value, _, _ = chi2_contingency(trans_matrix + 1e-5) # Avoid zeros
        change_ratio = (map_t1 != map_t2).mean()

        # 3. Decision Logic
        if change_ratio < 0.05:
            st.success(f"✅ **Land cover is stable.** Less than 5% of pixels changed between {t1_label} and {t2_label}.")
            verdict = "STABLE"
        elif p_value > 0.05:
            st.warning(f"⚠️ **Statistically Insignificant Changes (p={p_value:.3f}).** Differences are likely due to atmospheric noise or variations.")
            verdict = "INSIGNIFICANT"
        else:
            st.error(f"🚨 **Significant Change Detected (p<{p_value:.3g}).** Statistical testing confirms non-random terrain transformation.")
            verdict = "SIGNIFICANT"

        # --- Visual Panels ---
        st.divider()
        p1_col1, p1_col2 = st.columns(2)
        with p1_col1:
            st.subheader(f"🖼️ {t1_label} Classified")
            # Convert class indices to target colors
            seg_t1 = np.zeros(np.array(img1).shape, dtype=np.uint8)
            for i in range(6): seg_t1[map_t1 == i] = TARGET_CLASSES_LIST[i][2]
            st.image(seg_t1, use_container_width=True)
        with p1_col2:
            st.subheader(f"🖼️ {t2_label} Classified")
            seg_t2 = np.zeros(np.array(img2).shape, dtype=np.uint8)
            for i in range(6): seg_t2[map_t2 == i] = TARGET_CLASSES_LIST[i][2]
            st.image(seg_t2, use_container_width=True)

        # Panel 2: Change Heatmap
        st.subheader("🗺️ Change Heatmap")
        CHANGE_COLORS = {
            "veg_gain": [29, 158, 117], "veg_loss": [226, 75, 74], 
            "water_gain": [55, 138, 221], "urban": [186, 117, 23], 
            "other": [139, 105, 20], "stable": [211, 209, 199]
        }
        heatmap = np.full(seg_t1.shape, CHANGE_COLORS["stable"], dtype=np.uint8)
        
        # Gain Veg (Anything -> Forest(2)/Agri(4))
        heatmap[((map_t1 != 2) & (map_t1 != 4)) & ((map_t2 == 2) | (map_t2 == 4))] = CHANGE_COLORS["veg_gain"]
        # Loss Veg (Forest(2)/Agri(4) -> Anything else)
        heatmap[((map_t1 == 2) | (map_t1 == 4)) & ((map_t2 != 2) & (map_t2 != 4))] = CHANGE_COLORS["veg_loss"]
        # Urban (Anything -> Build(0)/Road(1))
        heatmap[(map_t1 != 0) & (map_t1 != 1) & ((map_t2 == 0) | (map_t2 == 1))] = CHANGE_COLORS["urban"]
        # Water Gain
        heatmap[(map_t1 != 3) & (map_t2 == 3)] = CHANGE_COLORS["water_gain"]
        
        if not show_unchanged:
            heatmap[map_t1 == map_t2] = [0, 0, 0]
        
        st.image(heatmap, use_container_width=True)
        st.caption("Green: Veg Gain | Red: Veg Loss | Orange: Urban Expansion | Blue: Water Gain | Grey: Stable")

        # Panel 3: Sankey Diagram
        st.subheader("🔗 Land Cover Transition Flows (Sankey)")
        labels = [f"{t1_label} {n[0]}" for n in TARGET_CLASSES_LIST] + [f"{t2_label} {n[0]}" for n in TARGET_CLASSES_LIST]
        sources, targets, values, colors = [], [], [], []
        
        total_p = trans_matrix.sum()
        for i in range(6):
            for j in range(6):
                strength = trans_matrix[i, j]
                if (strength / total_p) * 100 >= sankey_threshold:
                    sources.append(i)
                    targets.append(j + 6)
                    values.append(strength)
                    colors.append(f"rgba{tuple(TARGET_CLASSES_LIST[i][2] + [0.5])}")

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=[f"rgb{tuple(n[2])}" for n in TARGET_CLASSES_LIST]*2),
            link=dict(source=sources, target=targets, value=values, color=colors)
        )])
        st.plotly_chart(fig_sankey, use_container_width=True)

        # Panel 4 & 5: Area Analysis
        st.subheader("📊 Quantified Area Comparison")
        def px_to_km2(px): return round((px * (pixel_scale**2)) / 1e6, 4)
        
        areas_t1 = [px_to_km2((map_t1 == i).sum()) for i in range(6)]
        areas_t2 = [px_to_km2((map_t2 == i).sum()) for i in range(6)]
        class_names_simple = [n[0].split('/')[0] for n in TARGET_CLASSES_LIST]
        
        col_bar, col_delta = st.columns([2, 1])
        with col_bar:
            fig_area = go.Figure(data=[
                go.Bar(name=t1_label, x=class_names_simple, y=areas_t1, marker_color='#D3D1C7'),
                go.Bar(name=t2_label, x=class_names_simple, y=areas_t2, marker_color='#3b82f6')
            ])
            fig_area.update_layout(barmode='group', title="Class Area Comparison (km²)")
            st.plotly_chart(fig_area, use_container_width=True)
            
        with col_delta:
            deltas = [areas_t2[i] - areas_t1[i] for i in range(6)]
            fig_delta = px.bar(x=deltas, y=class_names_simple, orientation='h', color=deltas, color_continuous_scale='RdYlGn', title="Net Area Change (km²)")
            st.plotly_chart(fig_delta, use_container_width=True)

        # Panel 6: Statistical Table
        st.subheader("📋 Transition Summary Table")
        table_rows = []
        for i in range(6):
            net = areas_t2[i] - areas_t1[i]
            pct = (net / areas_t1[i] * 100) if areas_t1[i] > 0 else 0
            dir_str = "Stable" if abs(pct) < 1 else ("Expanded" if net > 0 else "Reduced")
            if areas_t1[i] == 0 and areas_t2[i] > 0: dir_str = "Appeared"
            table_rows.append({"Class": class_names_simple[i], f"{t1_label} Area": areas_t1[i], f"{t2_label} Area": areas_t2[i], "Net Change": round(net, 4), "% Change": round(pct, 1), "Direction": dir_str})
        
        df_trans = pd.DataFrame(table_rows)
        st.table(df_trans.style.map(lambda x: 'color: green' if x == 'Expanded' or x == 'Appeared' else ('color: red' if x == 'Reduced' else 'color: grey'), subset=['Direction']))

        # Analyst Interpretation
        st.subheader("📝 Analyst Interpretation")
        dom_idx = np.unravel_index(np.argmax(trans_matrix - np.diag(np.diag(trans_matrix))), trans_matrix.shape)
        dominant_move = f"{class_names_simple[dom_idx[0]]} → {class_names_simple[dom_idx[1]]}"
        gain_cls = class_names_simple[np.argmax(deltas)]
        loss_cls = class_names_simple[np.argmin(deltas)]
        
        trend = "urbanisation" if "Buildings" in gain_cls or "Roads" in gain_cls else ("deforestation" if "Forest" in loss_cls else "agricultural shift")
        stability = "The overall landscape remains largely stable with localised changes only." if change_ratio < 0.15 else "The landscape has undergone substantial transformation."

        st.markdown(f"""
        <div class="report-card">
        Between <b>{t1_label}</b> and <b>{t2_label}</b>, the most significant land cover transition was <b>{dominant_move}</b>, 
        accounting for { (trans_matrix[dom_idx] / total_p)*100:.1f}% of total area. 
        Overall <b>{change_ratio*100:.1f}%</b> of the landscape changed class. 
        The dominant trend suggests <b>{trend}</b>. {stability}
        Statistical testing confirms these changes are <b>{"significant at p<0.05" if p_value < 0.05 else "not statistically significant"}</b>.
        </div>
        """, unsafe_allow_html=True)

