#!/usr/bin/env python3
"""
asr_gradio_app.py - Live microphone demo for NVIDIA Nemotron ASR Streaming (0.6B)

A small Gradio web app: speak into the microphone and watch the
transcribed text grow live.

Start:
    pip install gradio soundfile        # once, inside the venv
    export HF_TOKEN="hf_xxxxxxxxxxxxx"   # optional, avoids HF rate limits
    python asr_gradio_app.py

Then open in your browser:  http://<server-ip>:7860
(or http://localhost:7860 when using an SSH tunnel)

Technical note:
    For maximum robustness this demo re-transcribes the accumulated audio
    buffer every ~0.5 s (offline transcribe on a growing buffer). This is
    version-independent and runs reliably. The model's true cache-aware
    streaming mechanism (conformer_stream_step) would be even lower latency,
    but is far more sensitive to the exact NeMo version.
"""

import os
import tempfile

import numpy as np
import torch
import torchaudio.functional as AF
import soundfile as sf
import gradio as gr

from huggingface_hub import login
import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
TARGET_SR = 16000          # model expects 16 kHz mono
MAX_SECONDS = 120          # safety limit for the buffer

# --- Optional HF login (prevents the rate-limit warning) ---
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)

# --- Load the model ONCE at startup ---
print(f"Loading model: {MODEL_NAME} ... (may take a while on first run)")
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
asr_model.eval()
print("Model loaded. Starting app ...")


def _to_mono_16k(sample_rate: int, samples: np.ndarray) -> np.ndarray:
    """Convert an audio chunk to float32 mono at 16 kHz."""
    audio = samples.astype(np.float32)

    # int16/int32 -> float [-1, 1]
    if np.issubdtype(samples.dtype, np.integer):
        max_val = np.iinfo(samples.dtype).max
        audio = audio / max_val

    # Stereo -> mono (average channels). Exactly the error from step 4/5!
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz if needed
    if sample_rate != TARGET_SR:
        tensor = torch.from_numpy(audio)
        tensor = AF.resample(tensor, sample_rate, TARGET_SR)
        audio = tensor.numpy()

    return audio.astype(np.float32)


def transcribe_stream(new_chunk, buffer):
    """Called for every incoming microphone chunk."""
    if buffer is None:
        buffer = np.zeros(0, dtype=np.float32)

    if new_chunk is not None:
        sr, data = new_chunk
        chunk = _to_mono_16k(sr, data)
        buffer = np.concatenate([buffer, chunk])

        # Cap the buffer at MAX_SECONDS (drop oldest samples)
        max_len = MAX_SECONDS * TARGET_SR
        if len(buffer) > max_len:
            buffer = buffer[-max_len:]

    # Only transcribe once there is something meaningful (> 0.3 s)
    if len(buffer) < int(0.3 * TARGET_SR):
        return "... (waiting for audio)", buffer

    # Write the buffer to a temporary WAV file and transcribe it
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, buffer, TARGET_SR, subtype="PCM_16")
        with torch.no_grad():
            output = asr_model.transcribe([tmp.name], verbose=False)

    result = output[0]
    text = getattr(result, "text", result)
    return text, buffer


def reset_state():
    """Reset the buffer and the text output."""
    return None, ""


with gr.Blocks(title="Nemotron ASR Streaming - Live Demo") as demo:
    gr.Markdown(
        "# NVIDIA Nemotron ASR Streaming - Live Demo\n"
        "Speak into the microphone - the text appears live. "
        "Use **Reset** to start a new recording."
    )

    state = gr.State(value=None)

    with gr.Row():
        mic = gr.Audio(
            sources=["microphone"],
            streaming=True,
            type="numpy",
            label="Microphone",
        )
        output = gr.Textbox(
            label="Transcript (live)",
            lines=8,
            placeholder="Your spoken text will appear here ...",
        )

    reset_btn = gr.Button("Reset")

    # Live streaming: a new chunk every 0.5 s; concurrency_limit=1
    # prevents overlapping GPU calls.
    mic.stream(
        fn=transcribe_stream,
        inputs=[mic, state],
        outputs=[output, state],
        stream_every=0.5,
        concurrency_limit=1,
    )

    reset_btn.click(fn=reset_state, inputs=None, outputs=[state, output])


if __name__ == "__main__":
    # server_name=0.0.0.0 -> reachable across the LAN (e.g. 192.168.2.x:7860)
    #demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
