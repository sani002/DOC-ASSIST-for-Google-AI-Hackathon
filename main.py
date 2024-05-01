import os
import streamlit as st
import librosa
import torch
import torchaudio
import numpy as np
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from dotenv import load_dotenv
import google.generativeai as genai
import time
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY) # Initialize the speech-to-text model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "bangla-speech-processing/BanglaASR"
model1 = genai.GenerativeModel('gemini-pro') # Set page width
st.set_page_config(
    page_title="Doc-Assist",
    page_icon="🎤",
    layout="wide",
) # Add custom CSS to set the font to Quicksand
st.markdown(
    """
    <style>
        .stApp, .stSelectbox, .stTextInput, .stTextArea, .stMultiSelect, .stSlider, .stCheckbox, .stRadio, .stButton>button {
            font-family: Quicksand !important;
        }
    </style>
    """,
    unsafe_allow_html=True
) # Layout for displaying content in two columns
col1, col2 = st.columns(2)
with col1:
    st.image("ASSIST.png", use_column_width=True)
    st.title('Voice Based Diagnostic AI Assistant')
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"], key="audio_uploader")
    st.write("## Raw Transcribed Text")

    response = None  # Initialize response outside the if block

    if uploaded_file is not None: # Temporarily save the uploaded file to process
        mp3_path = "temp_audio.wav"
        with open(mp3_path, "wb") as f:
            f.write(uploaded_file.getvalue()) # Perform speech-to-text on the saved audio file
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        tokenizer = WhisperTokenizer.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device) # Load and play the audio
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes, format="audio/wav") # Transcribe the audio and display text gradually
        with st.spinner("Transcribing Audio..."):
            speech_array, sampling_rate = torchaudio.load(mp3_path, format="mp3")
            speech_array = speech_array[0].numpy()
            speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate, target_sr=16000)
            input_features = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt").input_features # Split the transcription into smaller chunks
            predicted_ids = model.generate(inputs=input_features.to(device))[0]
            transcription = processor.decode(predicted_ids, skip_special_tokens=True)
            def stream_text():
                for word in transcription.split():
                    yield word + " "
                    time.sleep(0.05)
            str1 = "The Bengali string that is given here was transcribed from audio to text. It has errors here and there and lacks punctuation. Read it and try to understand what it is trying to say, then rewrite it properly in English in patient statement like: the patient is... (don't use any pronoun, write 'the patient' each time. don't change valuable numbers or medical details in the process):"
            st.write_stream(stream_text())
            result = f"{str1} {transcription}"
            response = model1.generate_content(result)
            
with st.spinner("Generating AI Response..."): # Send message to AI and display respons
    response_text = response.text if response else "Will be generated once the audio is processed..."
    with col2:
        st.image("ASSIST2.png", use_column_width=True)
        st.write("## Generated Patient Note") # Display mBot response gradually using st.write_stream
        def stream_mbot_response():
            for word in response_text.split():
                yield word + " "
                time.sleep(0.05)

        st.write_stream(stream_mbot_response())
        str2 = "In the given text some patient data and symptoms of disease is given. Generate a brief report of data, symptoms, possible three diagnosis and next steps (for each point write one line)."
        result1 = f"{str2} {response_text}"
        response1 = model1.generate_content(result1)

with st.spinner("Processing Diagnosis..."): # Diagnosis
    response_text1 = response1.text if response else "Will be generated once the audio is processed..."
    with col2:
        st.write("## Suggested Diagnosis")
        st.markdown(f"<p>{response_text1}</p>", unsafe_allow_html=True)