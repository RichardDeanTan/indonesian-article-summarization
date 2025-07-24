import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
from huggingface_hub import hf_hub_download
import requests

MODEL_NAME = "RichTan/indobart-custom"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
MIN_WORD_COUNT = 40

st.set_page_config(
    page_title="Indonesian News Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_tokenizer(model_name):
    try:
        with st.spinner("Loading model and tokenizer dari Hugging Face..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model dari Hugging Face: {str(e)}")
        return None, None

def load_training_config_from_hf(model_name):
    try:
        config_path = hf_hub_download(repo_id=model_name, filename="training_config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            return {
                "original_model": getattr(config, "original_model", "IndoBART"),
                "model_type": getattr(config, "model_type", "bart"),
                "vocab_size": getattr(config, "vocab_size", "Unknown")
            }
        except:
            return None

def generate_summary(text, model, tokenizer, max_length=64, num_beams=2):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        inputs = tokenizer(
            text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                do_sample=False
            )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()
    
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def main():
    st.title("📝 Indonesian News Summarizer")
    st.markdown("---")
    st.markdown("""
    Aplikasi Streamlit ini menggunakan fine-tuned **IndoBART** model untuk menghasilkan summary dari Text Artikel Indonesia. 
    Model ini di-load langsung dari Hugging Face Hub: `RichTan/indobart-custom`
    """)
    
    with st.container():
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    if model is None or tokenizer is None:
        st.error("Failed to load model from Hugging Face. Please check your internet connection and model accessibility.")
        st.info("💡 Make sure the model 'RichTan/indobart-custom' is publicly accessible on Hugging Face.")
        st.stop()
    
    training_config = load_training_config_from_hf(MODEL_NAME)
    
    with st.sidebar:
        st.header("📊 Model Information")
        
        st.info(f"""
        **🤗 Hugging Face Model:** `{MODEL_NAME}`  
        - **Language:** Indonesia  
        - **Original Model:** {training_config.get('original_model', 'N/A')}
        """)
        
        if training_config:
            if "test_rouge1" in training_config:
                st.info(f"""
                **Test Metrics:**
                - **ROUGE-1 Score:** {training_config.get('test_rouge1', 0):.4f}
                - **ROUGE-2 Score:** {training_config.get('test_rouge2', 0):.4f}
                - **ROUGE-L Score:** {training_config.get('test_rougeL', 0):.4f}
                """)
            else:
                st.info(f"""
                **Model Details:**
                - **Original Model:** {training_config.get('original_model', 'IndoBART')}
                - **Model Type:** {training_config.get('model_type', 'bart')}
                - **Vocab Size:** {training_config.get('vocab_size', 'Unknown')}
                """)
        
        st.markdown("---")
        st.header("⚙️ Generation Settings")
        
        max_length = st.slider(
            "Max Token Summary Length",
            min_value=100,
            max_value=200,
            value=MAX_TARGET_LENGTH,
            help="Maximum token length dari summary yang mau dibikin"
        )
        
        num_beams = st.slider(
            "Number of Beams",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of beams untuk beam search (higher ≈ better quality)"
        )

        st.markdown("---")
        st.header("💡 Tips for Better Results")

        st.info("""
        - Minimum panjang artikel: **40 words**
        - Menggunakan struktur bahasa Indonesia yang baik
        - Hindari penggunaan singkatan
        - Hindari penggunaan bahasa campuran (Indo-English)
        """)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📰 Input Article")
        
        input_option = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Use Example"],
            horizontal=True
        )
        
        if input_option == "Use Example":
            example_text = """Jakarta - Pemerintah Indonesia mengumumkan kebijakan baru untuk meningkatkan sektor pariwisata nasional. Menteri Pariwisata dan Ekonomi Kreatif menyatakan bahwa program ini akan fokus pada pengembangan destinasi wisata lokal dan peningkatan kualitas layanan. Kebijakan ini diharapkan dapat meningkatkan kunjungan wisatawan domestik dan mancanegara. Pemerintah juga akan memberikan insentif kepada pelaku usaha pariwisata untuk mendukung pemulihan ekonomi pasca pandemi. Program pelatihan untuk pemandu wisata dan pelaku UMKM di sektor pariwisata juga akan diintensifkan. Target pemerintah adalah meningkatkan kontribusi sektor pariwisata terhadap PDB nasional dalam tiga tahun ke depan."""
            
            input_text = st.text_area(
                "Indonesian Article Text:",
                value=example_text,
                height=200,
                help="Edit contoh text ini atau ubah dengan artikel Anda sendiri",
                placeholder="Masukkan artikel bahasa Indonesia di sini..."
            )
        else:
            input_text = st.text_area(
                "Indonesian Article Text:",
                height=200,
                placeholder="Masukkan artikel bahasa Indonesia di sini...",
                help="Paste your Indonesian article here"
            )

        if input_text:
            word_count = len(input_text.split())
            char_count = len(input_text)
            
            if word_count < MIN_WORD_COUNT:
                st.markdown(f'''
                <p style="color: orange;">⚠️ Current: {word_count} words. Minimal {MIN_WORD_COUNT} words untuk generate summary.</p>
                ''', unsafe_allow_html=True)
            else:
                st.caption(f"✅ Words: {word_count} | Characters: {char_count}")
        
        word_count = len(input_text.split()) if input_text else 0
        can_generate = input_text.strip() and word_count >= MIN_WORD_COUNT

        generate_button = st.button(
            "Generate Summary",
            type="primary",
            disabled=not can_generate,
            use_container_width=True
        )
    
    with col2:
        st.header("📋 Generated Summary")
        
        if generate_button and input_text.strip():
            with st.spinner("Generating summary..."):
                summary = generate_summary(
                    input_text,
                    model,
                    tokenizer,
                    max_length=max_length,
                    num_beams=num_beams
                )
                
                if summary:
                    st.text_area(
                        "Summary:",
                        value=summary,
                        height=150,
                        disabled=True
                    )
                    
                    summary_words = len(summary.split())
                    summary_chars = len(summary)
                    original_words = len(input_text.split())
                    compression_ratio = (len(input_text.split()) / summary_words) if summary_words > 0 else 0
                    compression_percentage = ((original_words - summary_words) / original_words * 100) if original_words > 0 else 0
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Words", summary_words)
                    with col_b:
                        st.metric("Characters", summary_chars)
                    with col_c:
                        st.metric("Compression", f"{compression_percentage:.2f}%")
                    
                    # Download button
                    st.download_button(
                        label="💾 Download Summary",
                        data=f"Original Article:\n{input_text}\n\nGenerated Summary:\n{summary}",
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                    
                    st.success("✅ Summary generation berhasil!")
                else:
                    st.error("❌ Failed summary generation.")
        
        elif not input_text.strip():
            st.info("Tolong tulis kata di input area untuk generate summary.")
        else:
            st.info("Click buttonnya untuk melihat hasil summary!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Richard Dean Tanjaya | Github: <a href="https://github.com/RichardDeanTan/indonesian-article-summarization" target="_blank">@RichardDeanTan</a></p>
        <p>Model: <a href="https://huggingface.co/RichTan/indobart-custom" target="_blank">🤗 RichTan/indobart-custom</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()