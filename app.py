"""
VoiceMorph - zero-shot voice conversion with a desktop GUI (school project).

Engine: OpenVoice v2 ToneColorConverter (MIT licence).
Flow:
  1. Calibrate: record ~10s of YOUR normal voice  -> source embedding
  2. Reference: upload video/audio of the target voice (audio is extracted
     with ffmpeg) -> target embedding
  3. Convert: speak/upload -> output sounds like the reference voice
  4. Live mode: converts mic chunks and plays them to a chosen output device
     (select a virtual cable to use it as a microphone in WhatsApp/Telegram)

Only use voices you own or have explicit permission to use.
"""
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import customtkinter as ctk
import numpy as np
import sounddevice as sd
import soundfile as sf
from tkinter import filedialog, messagebox

import imageio_ffmpeg

APP_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
CKPT = APP_DIR / "checkpoints_v2" / "converter"
WORK = Path(tempfile.mkdtemp(prefix="voicemorph_"))
SR = 22050
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ----------------------------------------------------------------- audio utils
def extract_audio(src: str, dst: Path) -> Path:
    """Pull audio from any video/audio file -> mono 22.05kHz wav."""
    cmd = [FFMPEG, "-y", "-i", src, "-vn", "-ac", "1", "-ar", str(SR), str(dst)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.run(cmd, check=True, capture_output=True, creationflags=flags)
    return dst


def record(seconds: float, device=None) -> np.ndarray:
    data = sd.rec(int(seconds * SR), samplerate=SR, channels=1,
                  dtype="float32", device=device)
    sd.wait()
    return data[:, 0]


# ---------------------------------------------------------------------- engine
class Engine:
    def __init__(self):
        import torch
        from openvoice.api import ToneColorConverter
        from openvoice import se_extractor
        self.torch = torch
        self.se_extractor = se_extractor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conv = ToneColorConverter(str(CKPT / "config.json"), device=self.device)
        self.conv.load_ckpt(str(CKPT / "checkpoint.pth"))
        self.src_se = None
        self.tgt_se = None

    def embed(self, wav_path: Path):
        se, _ = self.se_extractor.get_se(str(wav_path), self.conv, vad=True)
        return se

    def convert_file(self, in_wav: Path, out_wav: Path):
        self.conv.convert(audio_src_path=str(in_wav), src_se=self.src_se,
                          tgt_se=self.tgt_se, output_path=str(out_wav),
                          message="synthetic-voice")  # watermark tag


# ------------------------------------------------------------------------- GUI
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VoiceMorph")
        self.geometry("640x700")
        ctk.set_appearance_mode("dark")
        self.engine = None
        self.live = False
        self.input_wav = None
        self.output_wav = None
        self._build()
        threading.Thread(target=self._load_engine, daemon=True).start()

    # ---- layout
    def _build(self):
        pad = dict(padx=16, pady=6, fill="x")
        self.status = ctk.CTkLabel(self, text="Loading model...", anchor="w")
        self.status.pack(**pad)
        self.progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self.progress.pack(**pad)
        self.progress.start()

        self.consent = ctk.CTkCheckBox(
            self, text="I own this voice or have permission to clone it")
        self.consent.pack(**pad)

        box = ctk.CTkFrame(self); box.pack(**pad)
        ctk.CTkLabel(box, text="1. Your normal voice (calibration)").pack(anchor="w", padx=8, pady=4)
        ctk.CTkButton(box, text="Record 10s of my voice", command=self.calibrate).pack(padx=8, pady=4, fill="x")

        box = ctk.CTkFrame(self); box.pack(**pad)
        ctk.CTkLabel(box, text="2. Reference voice (video or audio)").pack(anchor="w", padx=8, pady=4)
        ctk.CTkButton(box, text="Upload reference...", command=self.pick_reference).pack(padx=8, pady=4, fill="x")
        self.ref_label = ctk.CTkLabel(box, text="none selected"); self.ref_label.pack(padx=8)

        box = ctk.CTkFrame(self); box.pack(**pad)
        ctk.CTkLabel(box, text="3. Convert").pack(anchor="w", padx=8, pady=4)
        row = ctk.CTkFrame(box, fg_color="transparent"); row.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(row, text="Upload audio", command=self.pick_input).pack(side="left", expand=True, fill="x", padx=2)
        ctk.CTkButton(row, text="Record 5s", command=self.record_input).pack(side="left", expand=True, fill="x", padx=2)
        ctk.CTkButton(box, text="Convert", command=self.convert).pack(padx=8, pady=4, fill="x")
        row = ctk.CTkFrame(box, fg_color="transparent"); row.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(row, text="Play result", command=self.play_result).pack(side="left", expand=True, fill="x", padx=2)
        ctk.CTkButton(row, text="Save .wav", command=self.save_result).pack(side="left", expand=True, fill="x", padx=2)

        box = ctk.CTkFrame(self); box.pack(**pad)
        ctk.CTkLabel(box, text="4. Live mode (calls)").pack(anchor="w", padx=8, pady=4)
        ins = [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
        outs = [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0]
        self.in_dev = ctk.CTkOptionMenu(box, values=ins or ["none"]); self.in_dev.pack(padx=8, pady=2, fill="x")
        self.out_dev = ctk.CTkOptionMenu(box, values=outs or ["none"]); self.out_dev.pack(padx=8, pady=2, fill="x")
        ctk.CTkLabel(box, text="Output -> pick 'CABLE Input' (VB-Cable), then set the app's mic to 'CABLE Output'",
                     wraplength=560, justify="left").pack(padx=8)
        self.live_btn = ctk.CTkButton(box, text="Start live", command=self.toggle_live)
        self.live_btn.pack(padx=8, pady=6, fill="x")

    # ---- helpers
    def say(self, msg):
        self.after(0, lambda: self.status.configure(text=msg))

    def _busy(self, on: bool):
        def f():
            (self.progress.start if on else self.progress.stop)()
        self.after(0, f)

    def _need_ready(self, need_src=True, need_tgt=True):
        if not self.consent.get():
            messagebox.showwarning("Consent", "Confirm you have permission to use this voice.")
            return False
        if self.engine is None:
            messagebox.showinfo("Wait", "Model is still loading.")
            return False
        if need_src and self.engine.src_se is None:
            messagebox.showinfo("Step 1", "Record your calibration voice first.")
            return False
        if need_tgt and self.engine.tgt_se is None:
            messagebox.showinfo("Step 2", "Upload a reference voice first.")
            return False
        return True

    def _load_engine(self):
        try:
            self.engine = Engine()
            self.say(f"Ready ({self.engine.device}).")
        except Exception as e:
            self.say(f"Model load failed: {e}")
        self._busy(False)

    def _bg(self, fn, label):
        def run():
            self._busy(True); self.say(label)
            try:
                fn(); self.say("Done.")
            except Exception as e:
                self.say(f"Error: {e}")
            self._busy(False)
        threading.Thread(target=run, daemon=True).start()

    # ---- actions
    def calibrate(self):
        if not self._need_ready(False, False):
            return
        def job():
            self.say("Speak naturally for 10 seconds...")
            audio = record(10)
            p = WORK / "me.wav"; sf.write(p, audio, SR)
            self.engine.src_se = self.engine.embed(p)
        self._bg(job, "Recording calibration...")

    def pick_reference(self):
        if not self._need_ready(False, False):
            return
        f = filedialog.askopenfilename(filetypes=[("Media", "*.mp4 *.mov *.mkv *.avi *.webm *.mp3 *.wav *.m4a *.ogg *.flac"), ("All", "*.*")])
        if not f:
            return
        def job():
            wav = extract_audio(f, WORK / "ref.wav")
            self.engine.tgt_se = self.engine.embed(wav)
            self.after(0, lambda: self.ref_label.configure(text=Path(f).name))
        self._bg(job, "Extracting audio and learning voice...")

    def pick_input(self):
        f = filedialog.askopenfilename(filetypes=[("Media", "*.mp4 *.mov *.mp3 *.wav *.m4a *.ogg *.flac"), ("All", "*.*")])
        if f:
            self.input_wav = extract_audio(f, WORK / "in.wav")
            self.say(f"Input: {Path(f).name}")

    def record_input(self):
        def job():
            audio = record(5)
            self.input_wav = WORK / "in.wav"; sf.write(self.input_wav, audio, SR)
        self._bg(job, "Recording 5s...")

    def convert(self):
        if not self._need_ready():
            return
        if self.input_wav is None:
            messagebox.showinfo("Input", "Upload or record something to convert.")
            return
        def job():
            self.output_wav = WORK / "out.wav"
            self.engine.convert_file(self.input_wav, self.output_wav)
        self._bg(job, "Converting...")

    def play_result(self):
        if self.output_wav and Path(self.output_wav).exists():
            data, sr = sf.read(self.output_wav)
            sd.play(data, sr)

    def save_result(self):
        if not self.output_wav:
            return
        f = filedialog.asksaveasfilename(defaultextension=".wav")
        if f:
            Path(f).write_bytes(Path(self.output_wav).read_bytes())

    # ---- live mode (chunked; expect ~1.5-3s latency with this engine)
    def toggle_live(self):
        if self.live:
            self.live = False
            self.live_btn.configure(text="Start live")
            return
        if not self._need_ready():
            return
        self.live = True
        self.live_btn.configure(text="Stop live")
        in_i = int(self.in_dev.get().split(":")[0])
        out_i = int(self.out_dev.get().split(":")[0])
        q_raw, q_out = queue.Queue(), queue.Queue()
        CH = 2.0

        def capture():
            while self.live:
                q_raw.put(record(CH, device=in_i))

        def process():
            n = 0
            while self.live:
                try:
                    chunk = q_raw.get(timeout=0.5)
                except queue.Empty:
                    continue
                if np.abs(chunk).max() < 0.01:       # silence gate
                    q_out.put((np.zeros_like(chunk), SR)); continue
                a, b = WORK / f"c{n % 2}.wav", WORK / f"o{n % 2}.wav"; n += 1
                sf.write(a, chunk, SR)
                self.engine.convert_file(a, b)
                q_out.put(sf.read(b))

        def play():
            while self.live:
                try:
                    data, sr = q_out.get(timeout=0.5)
                except queue.Empty:
                    continue
                sd.play(data, sr, device=out_i); sd.wait()

        for fn in (capture, process, play):
            threading.Thread(target=fn, daemon=True).start()
        self.say("Live: converting mic -> output device")


if __name__ == "__main__":
    App().mainloop()
