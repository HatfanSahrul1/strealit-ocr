import streamlit as st
import sys
import os
from PIL import Image
import numpy as np

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) 

if root_dir not in sys.path:
    sys.path.append(root_dir)

model_dir = os.path.join(current_dir, "model")
if model_dir not in sys.path:
    sys.path.append(model_dir)

try:
    from src.utility.preprocessing import ImagePreprocessor
    from src.model.florence import FlorenceModel
    from src.model.donut import DonutModel
except ImportError as e:
    st.error(f"Import Error: {e}. Cek struktur folder.")
    st.stop()

# --- CONFIG PAGE ---
st.set_page_config(layout="wide", page_title="Split Bill OCR Sat-Set", page_icon="🧾")

# --- CACHED FUNCTIONS ---

@st.cache_resource(show_spinner="Loading Florence-2 Model...")
def get_florence_model():
    return FlorenceModel()

@st.cache_resource(show_spinner="Loading Donut Model...")
def get_donut_model():
    return DonutModel()

@st.cache_data(show_spinner="Processing Image (Auto-Crop & Deskew)...")
def process_uploaded_image(image_file):
    return ImagePreprocessor.process_image(image_file)

# --- SESSION STATE ---
if 'receipt_data' not in st.session_state:
    st.session_state.receipt_data = None
if 'assignments' not in st.session_state:
    st.session_state.assignments = {} 

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    st.subheader("1. Partisipan")
    participants_input = st.text_area("Nama (pisahkan koma)", "Kamu, Teman 1, Teman 2")
    participants = [p.strip() for p in participants_input.split(',') if p.strip()]
    
    st.subheader("2. Model AI")
    model_choice = st.selectbox("Engine", ["Florence-2", "Donut"])

# --- MAIN PAGE ---
st.title("🧾 Split Bill OCR")
st.markdown("Upload nota, biarkan AI baca, lalu assign siapa yang makan.")

# SECTION 1: UPLOAD & PREPROCESS
st.subheader("📤 Upload Nota")
uploaded_file = st.file_uploader("Format: JPG, PNG, JPEG", type=['jpg', 'jpeg', 'png'])

processed_image = None

if uploaded_file is not None:
    try:
        processed_image = process_uploaded_image(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed_image, caption="AI Ready (Lurus & Bersih)", use_column_width=True)

        if st.button("SCAN MENU & HARGA", type="primary", use_container_width=True):
            if not participants:
                st.warning("Isi dulu daftar partisipan di sidebar!")
            else:
                try:
                    if model_choice == "Florence-2":
                        model = get_florence_model()
                    else:
                        model = get_donut_model()
                    
                    with st.spinner(f"Sedang membaca nota pake {model_choice}..."):
                        receipt = model.run(processed_image)
                        st.session_state.receipt_data = receipt
                        st.session_state.assignments = {} 
                    
                except Exception as e:
                    st.error(f"Model Crash: {e}")
                    
    except Exception as e:
         st.error(f"Gagal Preprocess: {e}")

# SECTION 2: ASSIGNMENT
if st.session_state.receipt_data:
    st.divider()
    st.subheader("📝 Assign Menu ke Partisipan")
    
    data = st.session_state.receipt_data
    
    if not data.items:
        st.warning("⚠️ Tidak ada item terdeteksi. Coba model lain atau foto ulang.")
    else:
        # Header Table
        h1, h2, h3, h4 = st.columns([3, 1, 2, 4])
        h1.markdown("**Menu**")
        h2.markdown("**Qty**")
        h3.markdown("**Total (Rp)**")
        h4.markdown("**Yang Makan**")
        
        updated_assignments = {}
        
        for item_id, item in data.items.items():
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 1, 2, 4])
                c1.text(item.name)
                c2.text(item.count)
                c3.text(f"{item.total_price:,.0f}")
                
                current = st.session_state.assignments.get(item_id, [])
                valid_current = [p for p in current if p in participants]
                
                selected = c4.multiselect(
                    f"Assign {item.name}",
                    options=participants,
                    default=valid_current,
                    label_visibility="collapsed",
                    key=f"item_{item_id}"
                )
                updated_assignments[item_id] = selected
        
        st.session_state.assignments = updated_assignments

        st.divider()
        
        # SECTION 3: TOTALAN
        st.subheader("$ Hasil Akhir")
        
        if st.button("Hitung Split Bill"):
            bill_summary = {p: 0.0 for p in participants}
            unassigned = []
            
            for item_id, item in data.items.items():
                assigned_people = st.session_state.assignments.get(item_id, [])
                
                if not assigned_people:
                    unassigned.append(item)
                else:
                    split = item.total_price / len(assigned_people)
                    for p in assigned_people:
                        bill_summary[p] += split
            
            import pandas as pd
            res1, res2 = st.columns(2)
            
            with res1:
                st.success("Tagihan Per Orang")
                df_bill = pd.DataFrame(list(bill_summary.items()), columns=["Nama", "Bayar (Rp)"])
                st.dataframe(df_bill, use_container_width=True, hide_index=True)
                
            with res2:
                if unassigned:
                    st.error(f"Belum Di-assign: Rp {sum(x.total_price for x in unassigned):,.0f}")
                    for x in unassigned:
                        st.caption(f"- {x.name} ({x.total_price:,.0f})")
                else:
                    st.info("Semua menu sudah bertuan!")
            
            st.metric("Total Nota Terbaca", f"Rp {data.total:,.0f}")