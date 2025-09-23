def assign_speakers(segments):
    """
    MVP: Segmentleri dönüşümlü Spk1/Spk2 etiketler.
    İleride pyannote/resemblyzer ile gerçek diarization'a geçilir.
    """
    out, spk = [], "Spk1"
    for s in segments:
        out.append({**s, "speaker": spk})
        spk = "Spk2" if spk == "Spk1" else "Spk1"
    return out
