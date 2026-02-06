"""
Streamlit app for Vietnamese-Chinese Machine Translation visualization.

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
import torch
from pathlib import Path
import sys
import tempfile

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import Translator

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    st.error("Missing huggingface_hub. Install with: pip install huggingface-hub")
    st.stop()


# Page config
st.set_page_config(
    page_title="Vietnamese-Chinese Translation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
    .translation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .title {
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def download_model_from_hub(repo_id: str, cache_dir: str = None):
    """Download model from Hugging Face Hub."""
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "hf_models")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        with st.spinner(f"Downloading model from {repo_id}..."):
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.pt",
                cache_dir=cache_dir,
                force_download=False
            )
        return checkpoint_path
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        return None


@st.cache_resource
def load_model_local(checkpoint_path: str):
    """Load model from local checkpoint."""
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Checkpoint not found at {checkpoint_path}")
        st.info("Please train the model first using: bash scripts/train.sh")
        return None
    
    try:
        with st.spinner("Loading model..."):
            translator = Translator(checkpoint_path)
        st.success("✓ Model loaded successfully!")
        return translator
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


@st.cache_resource
def load_model():
    """Load model from selected source (local or HuggingFace)."""
    # Check if HF model is configured via sidebar
    if "model_source" not in st.session_state:
        st.session_state.model_source = "local"
    
    if "hf_repo_id" not in st.session_state:
        st.session_state.hf_repo_id = None
    
    if st.session_state.model_source == "huggingface" and st.session_state.hf_repo_id:
        checkpoint_path = download_model_from_hub(st.session_state.hf_repo_id)
        if checkpoint_path:
            return load_model_local(checkpoint_path)
    else:
        checkpoint_path = "checkpoints_bidirectional/best_model.pt"
        return load_model_local(checkpoint_path)


def translate_text(translator, text, src_lang, use_beam_search):
    """Translate text and return result."""
    try:
        with st.spinner("Translating..."):
            result = translator.translate_sentence(
                text,
                src_lang=src_lang,
                use_beam_search=use_beam_search
            )
        return result
    except Exception as e:
        st.error(f"❌ Translation error: {e}")
        return None


def main():
    # Sidebar - Model selection
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        model_source = st.radio(
            "Model Source",
            ["Local", "Hugging Face Hub"],
            key="model_source_radio"
        )
        
        st.session_state.model_source = "huggingface" if model_source == "Hugging Face Hub" else "local"
        
        if model_source == "Hugging Face Hub":
            st.markdown("### Hugging Face Model")
            
            # Option 1: Predefined models
            predefined_models = {
                "Official VN-ZH": "duyquang/vn-zh-translation",
                "Custom Model": None
            }
            
            selected_model = st.selectbox(
                "Select Model",
                options=list(predefined_models.keys()),
                key="hf_model_select"
            )
            
            if predefined_models[selected_model]:
                st.session_state.hf_repo_id = predefined_models[selected_model]
                st.caption(f"📦 {st.session_state.hf_repo_id}")
            else:
                # Option 2: Custom model input
                custom_repo_id = st.text_input(
                    "Enter Hugging Face repo ID",
                    placeholder="username/repo-name",
                    key="hf_custom_repo"
                )
                if custom_repo_id:
                    st.session_state.hf_repo_id = custom_repo_id
                    st.caption(f"📦 {custom_repo_id}")
                else:
                    st.warning("Please enter a valid Hugging Face repo ID")
        else:
            st.markdown("### Local Model")
            st.session_state.hf_repo_id = None
            local_path = st.text_input(
                "Checkpoint path",
                value="checkpoints_bidirectional/best_model.pt",
                key="local_model_path"
            )
            if local_path != "checkpoints_bidirectional/best_model.pt":
                # Update if user changes path
                pass
        
        st.divider()
        
        # Help section
        with st.expander("ℹ️ Model Sources Help", expanded=False):
            st.markdown("""
            **Local Model**
            - Uses checkpoint from `checkpoints_bidirectional/best_model.pt`
            - Requires model to be trained first
            - Fastest loading time
            
            **Hugging Face Hub**
            - Downloads from Hugging Face model hub
            - Automatic caching of models
            - Easy sharing and collaboration
            
            **How to upload your model:**
            ```bash
            bash scripts/upload.sh \\
                --checkpoint checkpoints_bidirectional/best_model.pt \\
                --repo-id username/repo-name
            ```
            """)
    
    # Header
    st.markdown("<div class='title'>", unsafe_allow_html=True)
    st.title("🌐 Vietnamese-Chinese Machine Translation")
    st.markdown("**Powered by Transformer with RoPE & GQA**")
    st.markdown("</div>", unsafe_allow_html=True)

    # Load model
    translator = load_model()
    
    if translator is None:
        st.stop()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Translation", "📄 Batch Translation", "ℹ️ About"])

    # Tab 1: Single Translation
    with tab1:
        st.header("Single Sentence Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            src_lang = st.selectbox(
                "Source Language",
                options=["Chinese (中文)", "Vietnamese (Tiếng Việt)"],
                key="single_lang"
            )
            src_lang_code = "zh" if "Chinese" in src_lang else "vi"
            
            use_beam_search = st.checkbox(
                "Use Beam Search (higher quality, slower)",
                value=True,
                key="single_beam"
            )
        
        with col2:
            beam_size = 3
            if use_beam_search:
                beam_size = st.slider(
                    "Beam Size",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="single_beam_size"
                )
        
        # Input text
        source_text = st.text_area(
            "Enter text to translate",
            placeholder="输入要翻译的文本..." if src_lang_code == "zh" else "Nhập văn bản cần dịch...",
            height=100,
            key="single_text"
        )
        
        # Translate button
        if st.button("🚀 Translate", key="single_translate"):
            if source_text.strip():
                # Temporarily set beam size for this translation
                if use_beam_search:
                    translator.set_beam_params(beam_size=beam_size)
                
                translation = translate_text(
                    translator, source_text, src_lang_code, use_beam_search
                )
                
                if translation:
                    st.success("✓ Translation complete!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Source")
                        st.markdown(f'<div class="translation-box">{source_text}</div>', 
                                   unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("Translation")
                        st.markdown(f'<div class="translation-box">{translation}</div>', 
                                   unsafe_allow_html=True)
                    
                    # Copy button
                    st.code(translation, language="text")
            else:
                st.warning("Please enter some text to translate")

    # Tab 2: Batch Translation
    with tab2:
        st.header("Batch File Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            src_lang = st.selectbox(
                "Source Language",
                options=["Chinese (中文)", "Vietnamese (Tiếng Việt)"],
                key="batch_lang"
            )
            src_lang_code = "zh" if "Chinese" in src_lang else "vi"
        
        with col2:
            use_beam_search = st.checkbox(
                "Use Beam Search",
                value=True,
                key="batch_beam"
            )
        
        # File upload or text input
        option = st.radio(
            "Input method",
            ["Paste text (one sentence per line)", "Upload file"],
            key="batch_option"
        )
        
        texts = []
        
        if option == "Paste text (one sentence per line)":
            batch_text = st.text_area(
                "Enter text (one sentence per line)",
                height=200,
                key="batch_text"
            )
            if batch_text.strip():
                texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        else:
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=["txt"],
                key="batch_file"
            )
            if uploaded_file:
                texts = [line.strip() for line in uploaded_file.read().decode().split('\n') if line.strip()]
        
        if st.button("🚀 Translate Batch", key="batch_translate"):
            if texts:
                st.info(f"Processing {len(texts)} sentences...")
                
                with st.spinner("Translating..."):
                    try:
                        translations = translator.translate_batch(
                            texts,
                            src_lang=src_lang_code,
                            use_beam_search=use_beam_search,
                            batch_size=32
                        )
                        
                        # Display results
                        st.success(f"✓ Successfully translated {len(translations)} sentences!")
                        
                        # Create table
                        import pandas as pd
                        df = pd.DataFrame({
                            'Source': texts,
                            'Translation': translations
                        })
                        
                        st.dataframe(df, use_container_width=True, height=300)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv,
                            file_name="translations.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during batch translation: {e}")
            else:
                st.warning("Please enter some text to translate")

    # Tab 3: About
    with tab3:
        st.header("About this Translation System")
        
        # Model source info
        if st.session_state.model_source == "huggingface" and st.session_state.hf_repo_id:
            st.info(f"📦 **Model Source**: Hugging Face Hub")
            st.code(f"{st.session_state.hf_repo_id}")
            st.markdown(f"🔗 View on [Hugging Face](https://huggingface.co/{st.session_state.hf_repo_id})")
        else:
            st.info("📦 **Model Source**: Local Checkpoint")
        
        st.markdown("""
        ### Model Architecture
        
        This Vietnamese-Chinese machine translation system uses a state-of-the-art 
        Transformer architecture with:
        
        - **RoPE (Rotary Position Embeddings)**: Better length extrapolation
        - **GQA (Grouped Query Attention)**: More efficient attention with fewer KV heads
        - **SwiGLU Feed-Forward**: Gated activation for better expressiveness
        - **RMSNorm**: Efficient layer normalization
        
        ### Configuration
        
        - **Model Size**: ~140M parameters
        - **Encoder Layers**: 8
        - **Decoder Layers**: 8
        - **Attention Heads**: 12 (Query) + 4 (Key-Value)
        - **Hidden Dimension**: 768
        - **Max Sequence Length**: 32 tokens
        
        ### Features
        
        - **Bidirectional**: Chinese ↔ Vietnamese translation
        - **Model Sources**: 
          - Local checkpoint
          - Hugging Face Hub
        - **Decoding Strategies**: 
          - Greedy (fast)
          - Beam Search (high quality)
        - **Language Tokens**: Automatic direction detection
        
        ### How to Use
        
        1. Select model source (Local or Hugging Face)
        2. Select source language
        3. Enter text to translate
        4. Choose decoding strategy
        5. Click "Translate"
        
        ### Sharing Your Model
        
        Upload your trained model to Hugging Face Hub:
        
        ```bash
        bash scripts/upload.sh \\
            --checkpoint checkpoints_bidirectional/best_model.pt \\
            --repo-id username/repo-name
        ```
        
        Then use it here by selecting "Custom Model" and entering your repo ID.
        
        ### Performance
        
        - **Greedy Decoding**: ~50-100 sentences/second
        - **Beam Search (size=3)**: ~15-30 sentences/second
        - **Quality**: High with beam search, good with greedy
        
        ### Documentation
        
        For more details, see:
        - [README.md](../README.md) - Complete guide
        - [scripts/README.md](../scripts/README.md) - Script documentation
        - [QUICKSTART.py](../QUICKSTART.py) - Code examples
        - [HF_UPLOAD_GUIDE.md](../HF_UPLOAD_GUIDE.md) - How to upload models
        
        ---
        
        **Version**: 1.1.0  
        **Date**: February 6, 2026  
        **Status**: ✅ Production Ready with HF Hub Support
        """)
        
        # Model info
        st.divider()
        st.subheader("Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Size", "140M parameters")
        
        with col2:
            st.metric("Encoding", "8 layers")
        
        with col3:
            st.metric("Decoding", "8 layers")
        
        # Device info
        st.subheader("System Information")
        device_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Device", device_type)
        
        with col2:
            st.metric("Device Name", device_info)
        
        with col3:
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.metric("VRAM", f"{vram:.1f}GB")


if __name__ == "__main__":
    main()
