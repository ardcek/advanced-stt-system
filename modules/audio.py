import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

try:
    import msvcrt
except Exception:
    msvcrt = None

def record_stream(filename: str = "meeting.wav", fs: int = 16000, channels: int = 1):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        if status: print(status)
        q.put(indata.copy())

    print("Kayıt başladı… Durdurmak için [ENTER] ya da [q]. (Ctrl+C de olur)")
    with sf.SoundFile(filename, mode="w", samplerate=fs, channels=channels, subtype="PCM_16") as f:
        with sd.InputStream(samplerate=fs, channels=channels, dtype="int16", callback=cb):
            try:
                while True:
                    f.write(q.get())
                    if msvcrt and msvcrt.kbhit():
                        key = msvcrt.getwch()
                        if key in ("\r", "\n", "q", "Q"):
                            break
            except KeyboardInterrupt:
                pass
    print(f"Kayıt tamamlandı → {filename}")
    return filename

def record_audio(duration: int = 15, filename: str = "meeting.wav", fs: int = 16000, channels: int = 1):
    print(f"{duration} sn kayıt başlıyor…")
    data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype="int16")
    sd.wait()
    write(filename, fs, data)
    print(f"Kayıt tamamlandı → {filename}")
    return filename
