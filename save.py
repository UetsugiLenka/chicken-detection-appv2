# app.py
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Deteksi & Klasifikasi Daging Ayam",
    layout="wide"
)

# --- TITLE ---
st.title("Deteksi & Klasifikasi Daging Ayam")
st.markdown("Sistem Two-Stage Learning: YOLOv11 + ResNet50")
st.markdown("---")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    from ultralytics import YOLO
    import tensorflow as tf

    placeholder = st.empty()
    try:
        placeholder.info("Memuat model YOLOv11m...")
        detector = YOLO("models/yolo11m.pt")

        placeholder.info("Memuat model ResNet50...")
        classifier = tf.keras.models.load_model("models/best_resnet_momentum_0.3.keras")

        placeholder.success("Model berhasil dimuat.")
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

# --- HELPER FUNCTION ---
def classify_crop(crop):
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_resized = crop_pil.resize((224, 224))
    crop_array = np.array(crop_resized) / 255.0
    crop_array = np.expand_dims(crop_array, axis=0)
    pred_prob = classifier.predict(crop_array, verbose=0)[0]
    pred_label = class_names_resnet[np.argmax(pred_prob)]
    return pred_label

# --- SIDEBAR ---
st.sidebar.header("⚙️Pengaturan")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Keterangan Warna:**
- 🟢 Hijau  : Segar
- 🔴 Merah  : Busuk
""")

# --- TABS ---
tab1, tab2 = st.tabs(["Upload Gambar", "Kamera Live"])

# --- TAB 1: UPLOAD GAMBAR ---
with tab1:
    st.subheader("Unggah Gambar Ayam")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], key="upload")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Resize jika terlalu besar
        MAX_SIZE = 800
        w, h = image.size
        if w > MAX_SIZE or h > MAX_SIZE:
            scale = MAX_SIZE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            st.info(f"Gambar di-resize ke {new_w}x{new_h} untuk efisiensi.")

        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Mendeteksi bagian ayam..."):
            results = detector(image_cv, conf=confidence_threshold, imgsz=640, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            st.warning("Tidak ada ayam terdeteksi.")
        else:
            results_list = []
            img_with_boxes = image_cv.copy()

            with st.spinner("Mengklasifikasi kesegaran..."):
                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = map(int, box)
                    crop = img_with_boxes[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    pred_label = classify_crop(crop)
                    part_label = part_names_yolo[int(cls)]

                    results_list.append({
                        'Part Ayam': part_label,
                        'Kesegaran': pred_label
                    })

                    color = (0, 255, 0) if pred_label == 'segar' else (0, 0, 255)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_with_boxes, f"{part_label}: {pred_label}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Hasil Deteksi & Klasifikasi", use_container_width=True)

            if results_list:
                df = pd.DataFrame(results_list)
                st.dataframe(df, use_container_width=True)
                segar_count = len(df[df['Kesegaran'] == 'segar'])
                busuk_count = len(df[df['Kesegaran'] == 'busuk'])
                st.markdown(f"**Statistik:** Segar: {segar_count} | Busuk: {busuk_count}")
            else:
                st.warning("Tidak ada hasil klasifikasi.")

# --- TAB 2: KAMERA LIVE ---
with tab2:
    st.subheader("Deteksi Real-Time")
    st.info("Izinkan akses kamera dari browser.")

    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.detector = detector
            self.classifier = classifier
            self.conf_threshold = confidence_threshold
            self.frame_skip = 2
            self.frame_count = 0

        def recv(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            if self.frame_count % self.frame_skip != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            try:
                results = self.detector(img, conf=self.conf_threshold, imgsz=320, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                img_with_boxes = img.copy()

                for box, cls in zip(boxes, classes):
                    x1, y1, x2, y2 = map(int, box)
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    pred_label = classify_crop(crop)
                    part_label = part_names_yolo[int(cls)]
                    color = (0, 255, 0) if pred_label == 'segar' else (0, 0, 255)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img_with_boxes, f"{part_label}: {pred_label}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                return av.VideoFrame.from_ndarray(img_with_boxes, format="bgr24")
            except Exception as e:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live-camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --- FOOTER ---
st.markdown("---")
st.caption("Sistem Deteksi Kualitas Daging Ayam - Emkayn")