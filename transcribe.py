#!/usr/bin/env python3
"""
transcribe.py - Offline-Transkription mit NVIDIA Nemotron ASR Streaming (0.6B)

Aufruf:
    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"
    python transcribe.py audio.wav

Der HF-Token wird aus der Umgebungsvariable HF_TOKEN gelesen, damit er
nicht im Quellcode steht. Optional kann er auch per --token uebergeben werden.
"""

import os
import sys
import argparse

from huggingface_hub import login
import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"


def main():
    parser = argparse.ArgumentParser(
        description="Transkribiert eine Audiodatei mit Nemotron ASR Streaming."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default="audio.wav",
        help="Pfad zur Audiodatei (Mono-WAV). Standard: audio.wav",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging-Face-Token. Standard: Umgebungsvariable HF_TOKEN.",
    )
    args = parser.parse_args()

    # --- Hugging Face Login ---
    # Behebt die Warnung "You are sending unauthenticated requests to the HF Hub"
    if args.token:
        login(token=args.token)
        print("HF-Login erfolgreich (authentifizierte Downloads aktiv).")
    else:
        print(
            "WARNUNG: Kein HF-Token gefunden. Setze die Umgebungsvariable HF_TOKEN "
            "oder uebergib --token, um Rate-Limits zu vermeiden."
        )

    # --- Modell laden (wird beim ersten Mal heruntergeladen und gecacht) ---
    print(f"Lade Modell: {MODEL_NAME} ...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

    # --- Audiodatei transkribieren ---
    if not os.path.isfile(args.audio):
        sys.exit(f"FEHLER: Audiodatei nicht gefunden: {args.audio}")

    print(f"Transkribiere: {args.audio} ...")
    output = asr_model.transcribe([args.audio])

    # Neuere NeMo-Versionen geben Hypothesis-Objekte zurueck (.text),
    # aeltere liefern reine Strings. Wir fangen beides ab.
    result = output[0]
    text = getattr(result, "text", result)
    print("\n--- Transkript ---")
    print(text)


if __name__ == "__main__":
    main()
