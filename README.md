# VoiceMorph

Desktop app for zero-shot voice conversion, built with Python, PyTorch and [OpenVoice v2](https://github.com/myshell-ai/OpenVoice) (MIT).

## Features
- **Calibrate**: record ~10 s of your own voice to build a source voice embedding.
- **Reference voice**: upload any audio *or video* file; audio is extracted automatically with FFmpeg.
- **Convert**: upload or record speech and convert it to the reference voice, then play or save as `.wav`.
- **Live mode**: converts microphone audio in near real time and plays it to a chosen output device. Route it to a virtual cable (e.g. VB-Cable) to use it as a microphone in call apps.
- Consent checkbox. Only use voices you own or have permission to use.

## Tech
Python · PyTorch · OpenVoice v2 ToneColorConverter · CustomTkinter · sounddevice · FFmpeg

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Download the OpenVoice v2 checkpoints and place them so the converter is at `checkpoints_v2/converter/`, then run:
```bash
python app.py
```
