# app.py
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🐔 Deteksi & Klasifikasi Daging Ayam",
    layout="wide"
)

st.markdown("""
<style>
    .stWebRtcStreamer {
        max-width: 640px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

from PIL import Image
import cv2
import numpy as np
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# --- TITLE ---
st.title("🐔 Deteksi & Klasifikasi Daging Ayam")
st.markdown("### Sistem Two-Stage Learning: YOLOv11 + ResNet50")
st.markdown("---")

# --- LOAD MODELS (Cached) ---
@st.cache_resource
def load_models():
    from ultralytics import YOLO
    import tensorflow as tf
    from huggingface_hub import hf_hub_download
    import os

    placeholder = st.empty()
    try:
        placeholder.info("Memuat model YOLOv11m dari Hugging Face...")

        # Download YOLO
        yolo_path = hf_hub_download(
            repo_id="UetsugiLenka/yolreschicken",
            filename="yolo11m.pt"
        )
        detector = YOLO(yolo_path)

        placeholder.info("Memuat model ResNet50 dari Hugging Face...")

        # Download ResNet
        resnet_path = hf_hub_download(
            repo_id="UetsugiLenka/yolreschicken",
            filename="best_resnet_momentum_0.3.keras"
        )
        classifier = tf.keras.models.load_model(resnet_path)

        placeholder.success("Model berhasil dimuat dari Hugging Face!")
        return detector, classifier

    except Exception as e:
        placeholder.error(f"Gagal memuat model: {e}")
        return None, None

detector, classifier = load_models()
if detector is None or classifier is None:
    st.stop()

# --- CLASS NAMES ---
class_names_resnet = ['busuk', 'segar']
part_names_yolo = ['breast', 'leg', 'quarter', 'thigh', 'wing']

# --- SESSION STATE INITIALIZATION ---
if 'conf_threshold' not in st.session_state:
    st.session_state.conf_threshold = 0.5

# --- SIDEBAR ---
st.sidebar.header("📷 Pilihan Input")
input_option = st.sidebar.radio("Pilih sumber gambar:", ["Upload Gambar", "Kamera Live"])

st.sidebar.header("⚙️ Pengaturan")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Update nilai threshold ke session state
st.session_state.conf_threshold = confidence_threshold

st.sidebar.markdown("---")
st.sidebar.info("""
**Keterangan Warna:**
- 🟢 **Hijau**: Segar
- 🔴 **Merah**: Busuk
""")

# --- HELPER FUNCTION: CLASSIFY ---
def classify_crop(crop):
    try:
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_resized = crop_pil.resize((224, 224))
        crop_array = np.array(crop_resized) / 255.0
        crop_array = np.expand_dims(crop_array, axis=0)
        pred_prob = classifier.predict(crop_array, verbose=0)[0]
        pred_idx = np.argmax(pred_prob)
        pred_label = class_names_resnet[pred_idx]
        pred_conf = pred_prob[pred_idx]
        return pred_label, pred_conf
    except Exception as e:
        print(f"Error in classify_crop: {e}")
        return "error", 0.0

# --- UPLOAD IMAGE ---
if input_option == "Upload Gambar":
    st.subheader("Upload Gambar Ayam")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Resize
        MAX_SIZE = 800
        w, h = image.size
        if w > MAX_SIZE or h > MAX_SIZE:
            scale = MAX_SIZE / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            st.info(f"🖼️ Gambar di-resize ke {new_w}x{new_h} agar lebih cepat diproses.")

        st.image(image, caption="Gambar yang Diupload", use_container_width=True)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, image_cv)

        # Deteksi YOLO
        with st.spinner("🔍 Mendeteksi part ayam..."):
            results_yolo = detector(temp_path, conf=confidence_threshold, imgsz=640, verbose=False)
            boxes = results_yolo[0].boxes.xyxy.cpu().numpy()
            confs = results_yolo[0].boxes.conf.cpu().numpy()
            classes = results_yolo[0].boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            st.warning("❌ Tidak ada ayam terdeteksi.")
        else:
            results = []
            img_with_boxes = image_cv.copy()

            with st.spinner("Mengklasifikasi kesegaran..."):
                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)
                    crop = image_cv[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    pred_label, pred_conf = classify_crop(crop)
                    part_label = part_names_yolo[int(cls)]

                    results.append({
                        'Part Ayam': part_label,
                        'Kesegaran': pred_label,
                        'Confidence': pred_conf
                    })

                    color = (0, 255, 0) if pred_label == 'segar' else (0, 0, 255)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{part_label}: {pred_label} ({pred_conf:.2f})"
                    cv2.putText(img_with_boxes, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Hasil Deteksi & Klasifikasi", use_container_width=True)

            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                segar_count = len(df[df['Kesegaran']=='segar'])
                busuk_count = len(df[df['Kesegaran']=='busuk'])
                st.markdown(f"📊 **Statistik:** 🟢 Segar: {segar_count} | 🔴 Busuk: {busuk_count}")
            else:
                st.warning("❌ Tidak ada hasil klasifikasi.")

# --- KAMERA LIVE ---
# --- KAMERA LIVE ---
elif input_option == "Kamera Live":
    st.subheader("Deteksi Real-Time")
    st.info("Izinkan akses kamera dari browser (PC/HP).")

    # Update threshold ke session state
    st.session_state.conf_threshold = confidence_threshold

    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av

    # --- CLOSURE: Simpan model dan threshold di luar class ---
    detector_local = detector
    classifier_local = classifier
    conf_threshold_local = confidence_threshold

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.detector = detector_local
            self.classifier = classifier_local
            self.conf_threshold = conf_threshold_local
            self.frame_skip = 1
            self.frame_count = 0

        def classify_crop(self, crop):
            """Fungsi untuk klasifikasi dengan confidence"""
            try:
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop_resized = crop_pil.resize((224, 224))
                crop_array = np.array(crop_resized) / 255.0
                crop_array = np.expand_dims(crop_array, axis=0)
                pred_prob = self.classifier.predict(crop_array, verbose=0)[0]
                pred_idx = np.argmax(pred_prob)
                pred_label = class_names_resnet[pred_idx]
                pred_conf = pred_prob[pred_idx]
                return pred_label, pred_conf
            except Exception as e:
                print(f"Error in classify_crop: {e}")
                return "error", 0.0

        def recv(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            if self.frame_count % self.frame_skip != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            try:
                results = self.detector(img, conf=self.conf_threshold, imgsz=416, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                img_with_boxes = img.copy()

                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = map(int, box)
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Klasifikasi dengan confidence
                    pred_label, pred_conf = self.classify_crop(crop)
                    part_label = part_names_yolo[int(cls)]
                    color = (0, 255, 0) if pred_label == 'segar' else (0, 0, 255)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                    
                    # Tampilkan label dengan confidence
                    label_text = f"{part_label}: {pred_label} ({pred_conf:.2f})"
                    cv2.putText(img_with_boxes, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                return av.VideoFrame.from_ndarray(img_with_boxes, format="bgr24")
            except Exception as e:
                print(f"Error di VideoProcessor: {e}")
                return av.VideoFrame.from_ndarray(img, format="bgr24")

    # ✅ webrtc_streamer dengan resolusi tinggi

    
    webrtc_streamer(
        key="ayam-deteksi",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "min": 320},
                "height": {"ideal": 360, "min": 180},
                "frameRate": {"ideal": 30, "max": 30},
                "facingMode": "environment"  # kamera belakang
            },
            "audio": False,
        },
        async_processing=True,
    )

# --- FOOTER ---
st.markdown("---")
st.caption("Sistem Deteksi Kualitas Daging Ayam - Emkayn")
