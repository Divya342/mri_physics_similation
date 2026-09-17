import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="MRI Phantom & Physics Simulator", layout="wide")
st.title("🧠 Live MRI Image Contrast & Relaxation Physics Platform")
st.write("Select a preset weighting configuration or manually adjust sequence parameters below to observe changes.")

# --- FIX: Initialize the Slider Keys Directly ---
if "tr_slider" not in st.session_state:
    st.session_state.tr_slider = 500
if "te_slider" not in st.session_state:
    st.session_state.te_slider = 15
if "fa_slider" not in st.session_state:
    st.session_state.fa_slider = 60

# --- FIX: Buttons update the slider state directly ---
st.subheader("🎯 Contrast Quick Presets")
p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    if st.button("🔴 T1-Weighted Preset", use_container_width=True):
        st.session_state.tr_slider = 400
        st.session_state.te_slider = 10
        st.session_state.fa_slider = 80

with p_col2:
    if st.button("🟢 T2-Weighted Preset", use_container_width=True):
        st.session_state.tr_slider = 3000
        st.session_state.te_slider = 100
        st.session_state.fa_slider = 15

with p_col3:
    if st.button("🔵 Proton Density (PD) Preset", use_container_width=True):
        st.session_state.tr_slider = 3000
        st.session_state.te_slider = 10
        st.session_state.fa_slider = 15

with p_col4:
    if st.button("🔄 Reset to Default", use_container_width=True):
        st.session_state.tr_slider = 500
        st.session_state.te_slider = 15
        st.session_state.fa_slider = 60
st.write("---")

# 2. Sidebar Parameters (Letting Streamlit manage 'value' entirely via the 'key')
st.sidebar.header("🎛️ Pulse Sequence Parameters")
tr = st.sidebar.slider("Repetition Time (TR) in ms", min_value=10, max_value=4000, key="tr_slider", step=50)
te = st.sidebar.slider("Echo Time (TE) in ms", min_value=2, max_value=200, key="te_slider", step=2)
fa_deg = st.sidebar.slider("Flip Angle (α) in Degrees", min_value=5, max_value=180, key="fa_slider", step=5)

# Convert flip angle to radians for physics calculations
fa_rad = np.radians(fa_deg)

# 3. Define Tissue Physics Matrix Constants (T1, T2, PD at 3T)
tissue_props = {
    1: [850, 80, 0.70],  # White Matter (WM)
    2: [1350, 110, 0.80],  # Gray Matter (GM)
    3: [4000, 2000, 1.00],  # Cerebrospinal Fluid (CSF)
    0: [1, 1, 0.0]  # Background (Air)
}
tissue_names = {1: "White Matter", 2: "Gray Matter", 3: "CSF"}
colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}


# 4. Generate an Anatomically-Inspired Synthetic Spatial Phantom Matrix
@st.cache_data
def generate_base_phantom():
    grid_size = 128
    phantom = np.zeros((grid_size, grid_size))
    x, y = np.ogrid[-grid_size / 2:grid_size / 2, -grid_size / 2:grid_size / 2]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    head_contour = (grid_size * 0.43) * (1.0 - 0.05 * np.abs(np.cos(theta)) + 0.04 * np.cos(2 * theta))
    phantom[r < head_contour] = 3

    gm_contour = head_contour - 4
    phantom[r < gm_contour] = 2

    wm_contour = (grid_size * 0.28) * (1.0 + 0.08 * np.sin(4 * theta) - 0.03 * np.cos(6 * theta))
    phantom[r < wm_contour] = 1

    v_left_horn = (((x + 5) - 0.3 * y) ** 2 / 14 ** 2) + ((y + 10) ** 2 / 6 ** 2) < 1
    v_right_horn = (((x + 5) + 0.3 * y) ** 2 / 14 ** 2) + ((y - 10) ** 2 / 6 ** 2) < 1
    v_center = ((x - 8) ** 2 / 6 ** 2) + (y ** 2 / 4 ** 2) < 1
    phantom[v_left_horn | v_right_horn | v_center] = 3

    fissure = (np.abs(y) < 1.5) & (r < gm_contour) & ~(r < wm_contour - 10)
    phantom[fissure] = 2

    return phantom


base_phantom = generate_base_phantom()

# 5. Math Engine: Apply Steady-State Gradient-Echo Equation to each pixel
@st.cache_data(show_spinner=False)
def calculate_mri_image(phantom, tr, te, fa_rad):
    image_out = np.zeros_like(phantom, dtype=float)

    for label, constants in tissue_props.items():
        if label == 0:
            continue
        t1, t2, pd = constants

        e1 = np.exp(-tr / t1)
        e2 = np.exp(-te / t2)

        numerator = pd * np.sin(fa_rad) * (1 - e1)
        denominator = 1 - (np.cos(fa_rad) * e1)

        signal = (numerator / denominator) * e2
        image_out[phantom == label] = signal

    return image_out


simulated_image = calculate_mri_image(base_phantom, tr, te, fa_rad)

# 6. UI Layout - Split into Two Columns
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🖥️ Reconstructed Image Slice")

    fig_img, ax_img = plt.subplots(figsize=(6, 6), facecolor='black')
    vmax_val = max(0.1, np.max(simulated_image))
    ax_img.imshow(simulated_image, cmap='gray', vmin=0, vmax=vmax_val)
    ax_img.axis('off')
    st.pyplot(fig_img)

    # Classification logic display
    st.write("---")
    if tr <= 600 and fa_deg >= 45 and te <= 25:
        st.success("**Contrast Profile: T1-Weighted** (White Matter is bright, Gray Matter is intermediate, CSF is dark)")
    elif tr >= 2000 and fa_deg <= 25 and te >= 60:
        st.success("**Contrast Profile: T2-Weighted** (CSF/Fluid features are bright, Brain tissue is darker)")
    elif tr >= 2000 and fa_deg <= 25 and te <= 25:
        st.success("**Contrast Profile: Proton Density (PD)** (Contrast mapped purely by basic tissue hydrogen density)")
    else:
        st.info("**Contrast Profile: Mixed Steady-State Mode**")

with col2:
    st.subheader("📈 Signal Relaxation Physics Dynamics")

    time_t1 = np.linspace(0, 4000, 400)
    time_t2 = np.linspace(0, 250, 250)

    fig_curves, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7.5))

    # Plot Longitudinal Relaxation (T1 Recovery curves)
    for label, constants in tissue_props.items():
        if label == 0: continue
        t1, _, pd = constants
        e1_t = np.exp(-time_t1 / t1)
        curve_val = pd * np.sin(fa_rad) * (1 - e1_t) / (1 - np.cos(fa_rad) * e1_t)
        ax1.plot(time_t1, curve_val, label=tissue_names[label], color=colors[label], lw=2)

    ax1.axvline(x=tr, color='red', linestyle='--', alpha=0.7, label=f'Current TR ({tr}ms)')
    ax1.set_title("Longitudinal Recovery (T1 Profile with Flip Angle Interaction)")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Available Signal Vector")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.2)

    # Plot Transverse Relaxation (T2 Decay curves)
    for label, constants in tissue_props.items():
        if label == 0: continue
        t1, t2, pd = constants
        e1 = np.exp(-tr / t1)
        initial_mxy = pd * np.sin(fa_rad) * (1 - e1) / (1 - np.cos(fa_rad) * e1)
        ax2.plot(time_t2, initial_mxy * np.exp(-time_t2 / t2), label=tissue_names[label], color=colors[label], lw=2)

    ax2.axvline(x=te, color='red', linestyle='--', alpha=0.7, label=f'Current TE ({te}ms)')
    ax2.set_title("Transverse Signal Decay (T2 Profile)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Transverse Echo Signal")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig_curves)
