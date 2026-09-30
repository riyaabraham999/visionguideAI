import streamlit as st
import cv2
import numpy as np
from PIL import Image

from age_gender import analyze_age_gender
from pose_estimation import detect_pose
from image_caption import generate_caption

# Page configuration
st.set_page_config(
    page_title="AI Human Analysis System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Professional Enterprise Look
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Section */
.hero-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(147, 51, 234, 0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 18px;
    padding: 26px 30px;
    margin-bottom: 24px;
}

.hero-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4f46e5;
    background: rgba(99, 102, 241, 0.12);
    padding: 4px 12px;
    border-radius: 9999px;
    margin-bottom: 12px;
    border: 1px solid rgba(99, 102, 241, 0.25);
}

.hero-header {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0 0 8px 0;
    background: linear-gradient(135deg, #1e293b 0%, #3b82f6 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

@media (prefers-color-scheme: dark) {
    .hero-header {
        background: linear-gradient(135deg, #f8fafc 0%, #60a5fa 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
}

.hero-subtitle {
    font-size: 0.98rem;
    color: #64748b;
    margin: 0 0 16px 0;
    max-width: 800px;
    line-height: 1.5;
}

.tag-badge {
    display: inline-flex;
    align-items: center;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(148, 163, 184, 0.14);
    color: #475569;
    border: 1px solid rgba(148, 163, 184, 0.22);
    margin-right: 8px;
    margin-top: 4px;
}

@media (prefers-color-scheme: dark) {
    .tag-badge {
        color: #cbd5e1;
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.12);
    }
}

/* Feature Cards on landing */
.feature-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 16px;
    padding: 24px 20px;
    height: 100%;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
}

.feature-icon-wrapper {
    font-size: 2rem;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 1.12rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.feature-desc {
    font-size: 0.88rem;
    color: #64748b;
    line-height: 1.55;
    margin: 0;
}

/* Meta pill */
.meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 10px;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.22);
    margin-bottom: 16px;
}

/* Caption box */
.caption-quote {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(168, 85, 247, 0.08) 100%);
    border-left: 4px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    font-size: 1.05rem;
    font-weight: 500;
    line-height: 1.6;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 18px 0;">
        <div style="font-size: 2.2rem; margin-bottom: 6px;">🤖</div>
        <div style="font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em;">VisionPulse AI</div>
        <div style="font-size: 0.78rem; color: #64748b; font-weight: 500;">Human Analysis System</div>
    </div>
    """, unsafe_allow_html=True)

    module = st.selectbox(
        "Select Analysis Module",
        ["All Modules", "Human Pose Estimation", "Gender & Age Detection", "Image Caption Generator"]
    )

    st.markdown("---")

    st.markdown("#### ⚡ Active Pipeline")
    if module == "All Modules":
        st.info("Running full multi-modal pipeline: Demographics + Pose + Scene Captioning.")
    elif module == "Human Pose Estimation":
        st.info("Tracking 33 skeletal keypoints and identifying bodily posture and gestures.")
    elif module == "Gender & Age Detection":
        st.info("Detecting faces, predicting gender, and estimating age ranges.")
    elif module == "Image Caption Generator":
        st.info("Generating descriptive natural-language context using BLIP vision transformer.")

    st.markdown("---")
    with st.expander("ℹ️ System Diagnostics", expanded=False):
        st.caption("• **Pose Detector:** MediaPipe Pose (33 Landmarks)")
        st.caption("• **Demographics:** OpenCV DNN (Caffe)")
        st.caption("• **Captioning:** BLIP Vision-Language")
        st.caption("• **Execution:** Python 3.11 / Streamlit")

# Hero Header
st.markdown("""
<div class="hero-box">
    <div class="hero-pill">Next-Gen Computer Vision</div>
    <h1 class="hero-header">AI-Based Human Analysis System</h1>
    <p class="hero-subtitle">
        An integrated multi-modal platform for automated face & demographic classification, 
        skeletal human pose identification, and deep vision-language scene understanding.
    </p>
    <div>
        <span class="tag-badge">🧍 MediaPipe Pose</span>
        <span class="tag-badge">👤 Age & Gender DNN</span>
        <span class="tag-badge">📝 BLIP Transformers</span>
        <span class="tag-badge">⚡ Real-Time Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an image for analysis", 
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w = image_rgb.shape[:2]

    # File info banner
    st.markdown(f"""
    <div class="meta-chip">
        <span>📸 <strong>{uploaded_file.name}</strong></span>
        <span>•</span>
        <span>📐 {img_w} × {img_h} px</span>
        <span>•</span>
        <span>🔍 Module: <em>{module}</em></span>
    </div>
    """, unsafe_allow_html=True)

    # 1. Standalone Human Pose Estimation Module
    if module == "Human Pose Estimation":
        st.subheader("🧍 Human Pose Estimation")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📷 Original Image")
            st.image(image_rgb, use_container_width=True)
        with col2:
            st.markdown("##### ✨ Pose Estimation")
            pose_image, detected, pose_info = detect_pose(image_bgr)
            pose_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            st.image(pose_rgb, use_container_width=True)
            if detected:
                st.success(f"Pose: {pose_info['pose_name']}")
            else:
                st.warning("No human pose detected.")

    # 2. Standalone Gender & Age Detection Module
    elif module == "Gender & Age Detection":
        st.subheader("👤 Gender & Age Detection")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📷 Original Image")
            st.image(image_rgb, use_container_width=True)
        with col2:
            st.markdown("##### ✨ Age/Gender Analysis")
            result, count, details = analyze_age_gender(image_bgr)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, use_container_width=True)
            st.write(f"**Faces detected:** {count}")
            if count == 0:
                st.warning("No face detected.")
            else:
                for idx, d in enumerate(details, start=1):
                    icon = "👨" if d["gender"] == "Male" else "👩"
                    st.success(f"{icon} **Face {idx}:** {d['gender']}, Age {d['age']}")

    # 3. Standalone Image Caption Generator Module
    elif module == "Image Caption Generator":
        st.subheader("📝 Image Caption Generator")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📷 Original Image")
            st.image(image_rgb, use_container_width=True)
        with col2:
            st.markdown("##### ✨ Generated Description")
            with st.spinner("Generating caption with BLIP..."):
                caption = generate_caption(image_rgb)
            st.markdown(f"""
            <div class="caption-quote">
                💬 <em>"{caption}"</em>
            </div>
            """, unsafe_allow_html=True)
            st.success("Caption generated successfully.")

    # 4. All Modules Selected (Tabbed Multi-Modal Dashboard)
    else:
        tab_pose, tab_face, tab_caption = st.tabs([
            "🧍 Human Pose Estimation",
            "👤 Gender & Age Detection",
            "📝 Image Caption Generator"
        ])

        with tab_pose:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📷 Original Image")
                st.image(image_rgb, use_container_width=True)
            with col2:
                st.markdown("##### ✨ Pose Estimation")
                pose_image, detected, pose_info = detect_pose(image_bgr)
                pose_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
                st.image(pose_rgb, use_container_width=True)
                if detected:
                    st.success(f"Pose: {pose_info['pose_name']}")
                else:
                    st.warning("No human pose detected.")

        with tab_face:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📷 Original Image")
                st.image(image_rgb, use_container_width=True)
            with col2:
                st.markdown("##### ✨ Age/Gender Analysis")
                result, count, details = analyze_age_gender(image_bgr)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)
                st.write(f"**Faces detected:** {count}")
                if count == 0:
                    st.warning("No face detected.")
                else:
                    for idx, d in enumerate(details, start=1):
                        icon = "👨" if d["gender"] == "Male" else "👩"
                        st.success(f"{icon} **Face {idx}:** {d['gender']}, Age {d['age']}")

        with tab_caption:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 📷 Original Image")
                st.image(image_rgb, use_container_width=True)
            with col2:
                st.markdown("##### ✨ Generated Description")
                with st.spinner("Generating caption with BLIP..."):
                    caption = generate_caption(image_rgb)
                st.markdown(f"""
                <div class="caption-quote">
                    💬 <em>"{caption}"</em>
                </div>
                """, unsafe_allow_html=True)
                st.success("Caption generated successfully.")

else:
    # Landing Page Feature Showcase
    st.markdown("### 🚀 Available Capabilities")
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">🧍</div>
            <div class="feature-title">Pose & Posture Identification</div>
            <p class="feature-desc">
                Tracks 33 full-body landmark coordinates with MediaPipe. Identifies postures (standing, sitting, squatting, bending, reclining) and arm gestures in real time.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">👤</div>
            <div class="feature-title">Gender & Age Estimation</div>
            <p class="feature-desc">
                Applies deep convolutional neural networks to detect human faces, estimate age demographics, and classify gender with confidence filtering.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with f_col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">📝</div>
            <div class="feature-title">Vision-Language Captioning</div>
            <p class="feature-desc">
                Employs Salesforce BLIP conditional generation to produce natural-language descriptions of humans, actions, and environmental context.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👆 **Get Started:** Drag and drop or browse a photo above to run the analysis.")
