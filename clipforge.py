#!/usr/bin/env python3
"""
ClipForge v2 - self-hosted AI video clipper. Unlimited, no credits, no watermark.

Features
  Input      YouTube/any URL (yt-dlp), file upload, whole channel/playlist auto-import (+ watch mode)
  AI         Whisper transcription, audio-energy + visual-motion analysis, text hook scoring,
             optional local-LLM (Ollama) ranking, virality score, title/description/hashtags
  Video      face-tracking smooth reframe, 9:16 / 1:1 / 4:5 / 16:9, auto B-roll insert from your folder
  Captions   animated word-by-word captions, several styles/positions, translated captions
  Dubbing    translate + free neural TTS (edge-tts), ducked original audio
  Projects   saved forever, transcript kept, trim/re-render any clip, posts.csv publish pack
  Publish    YouTube Shorts upload via your own Google OAuth

CLI:  python clipforge.py "URL" --clips 8
      python clipforge.py --channel "https://youtube.com/@name" --latest 3 --watch 3600
      python clipforge.py video.mp4 --lang es --dub --broll-dir ./broll
Web:  python clipforge.py --serve
"""
import argparse, asyncio, csv, hmac, json, os, re, shutil, subprocess, sys, tempfile, threading, time
import urllib.request, uuid
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

OUT_ROOT = Path(os.getenv("CLIPFORGE_OUTPUT_DIR", "output")); OUT_ROOT.mkdir(parents=True, exist_ok=True)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ASPECTS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "4:5": (1080, 1350), "16:9": (1920, 1080)}

STOP = set("""a an the and or but if so to of in on at for with from by is are was were be been it this that
these those i you he she we they me him her us them my your his our their as not no yes do does did have has
had will would can could should just like really very about than then there here what when where who why how
um uh yeah okay oh gonna wanna kind sort mean know think because going thing things""".split())
HOOK_WORDS = set("""secret never always mistake truth biggest worst best shocking crazy insane unbelievable
nobody everyone why how stop warning actually honestly money free hack lesson failed won lost changed
important problem story remember imagine""".split())

STYLES = {  # p=highlight, s=base colour (BGR ASS), bs=BorderStyle
    "yellow":  dict(p="&H0000F0FF", s="&H00FFFFFF", size=84, upper=True, karaoke=True, bs=1),
    "green":   dict(p="&H0000FF00", s="&H00FFFFFF", size=84, upper=True, karaoke=True, bs=1),
    "pink":    dict(p="&H00B469FF", s="&H00FFFFFF", size=84, upper=True, karaoke=True, bs=1),
    "minimal": dict(p="&H00FFFFFF", s="&H00FFFFFF", size=62, upper=False, karaoke=False, bs=1),
    "boxed":   dict(p="&H00FFFFFF", s="&H00FFFFFF", size=70, upper=False, karaoke=False, bs=3),
}
VOICES = {"en": "en-US-AriaNeural", "es": "es-ES-ElviraNeural", "fr": "fr-FR-DeniseNeural",
          "de": "de-DE-KatjaNeural", "pt": "pt-BR-FranciscaNeural", "it": "it-IT-ElsaNeural",
          "ja": "ja-JP-NanamiNeural", "ko": "ko-KR-SunHiNeural", "zh": "zh-CN-XiaoxiaoNeural",
          "hi": "hi-IN-SwaraNeural", "id": "id-ID-GadisNeural", "ru": "ru-RU-SvetlanaNeural",
          "ar": "ar-SA-ZariyahNeural", "tl": "fil-PH-BlessicaNeural", "fil": "fil-PH-BlessicaNeural"}


@dataclass
class Opts:
    n: int = 8
    min_len: int = 20
    max_len: int = 60
    model: str = "small"
    source_lang: str = ""
    captions: bool = True
    style: str = "yellow"
    position: str = "bottom"      # bottom|middle|top
    words_per_card: int = 3
    aspect: str = "9:16"
    track: bool = True
    use_llm: bool = True
    lang: str = ""                # target language code (captions translated / dubbed)
    dub: bool = False
    voice: str = ""
    broll_dir: str = ""
    llm_model: str = ""


# ------------------------------------------------------------------ helpers
def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-1500:]}")
    return r.stdout


def need(tool):
    if not shutil.which(tool):
        raise RuntimeError(f"'{tool}' not found. Install it first (see README).")


def duration_of(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", str(path)]).strip())


def probe(path):
    d = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)]))
    s = d["streams"][0]
    a, b = (s["r_frame_rate"].split("/") + ["1"])[:2]
    fps = float(a) / float(b or 1) or 30.0
    return {"w": s["width"], "h": s["height"], "fps": round(fps, 3), "dur": float(d["format"]["duration"])}


def download(url, workdir):
    need("yt-dlp")
    run(["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b", "--no-playlist",
         "--merge-output-format", "mp4", "-o", str(workdir / "source.%(ext)s"), url])
    return next(workdir.glob("source.*"))


def list_channel(url, n):
    need("yt-dlp")
    out = run(["yt-dlp", "--flat-playlist", "--playlist-end", str(n), "--print", "%(url)s", url])
    return [l.strip() for l in out.splitlines() if l.strip().startswith("http")]


# --------------------------------------------------------------- local LLM
def llm(prompt, timeout=180):
    """Free local LLM through Ollama. Returns dict or None (everything has a non-LLM fallback)."""
    try:
        body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json",
                           "options": {"temperature": 0.2, "num_ctx": 8192}}).encode()
        req = urllib.request.Request(OLLAMA_URL + "/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(json.loads(r.read())["response"])
    except Exception:
        return None


# ------------------------------------------------------------ transcription
def transcribe(video, o, progress=print):
    from faster_whisper import WhisperModel
    progress(f"Transcribing with Whisper '{o.model}' (first run downloads the model)...")
    model = WhisperModel(o.model, device="auto", compute_type="auto")
    segs, info = model.transcribe(str(video), word_timestamps=True, vad_filter=True,
                                  language=o.source_lang or None)
    out = []
    for s in segs:
        words = [{"w": w.word.strip(), "s": w.start, "e": w.end} for w in (s.words or []) if w.word.strip()]
        if words:
            out.append({"text": s.text.strip(), "start": s.start, "end": s.end, "words": words})
    return out, info.language


# ------------------------------------------------- audio / visual analysis
def audio_energy(video):
    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "8000",
                        "-f", "s16le", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32)
    if a.size < 8000:
        return np.zeros(1)
    secs = a[: a.size // 8000 * 8000].reshape(-1, 8000)
    rms = np.sqrt((secs ** 2).mean(1))
    return rms / (rms.max() + 1e-6)


def visual_motion(video, step=2.0):
    import cv2, numpy as np
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    vals, prev, t = [], None, 0.0
    while t < total:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        vals.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev, t = g, t + step
    cap.release()
    v = np.array(vals or [0.0])
    return v / (v.max() + 1e-6), step


def window_mean(arr, step, a, b):
    seg = arr[int(a / step): int(b / step) + 1]
    return float(seg.mean()) if len(seg) else 0.0


# ------------------------------------------------------- highlight finding (v3 engine)
# Stage 1  build real sentences from word timestamps
# Stage 2  enumerate every possible clip window, score with a 12-signal heuristic (text + audio + visual)
# Stage 3  (Ollama) LLM reads the WHOLE transcript chunk by chunk and nominates moments like an editor
# Stage 4  boundary refinement: trim weak openers, extend dependent starts, end on a completed thought
# Stage 5  (Ollama) 6-dimension rubric rating of each finalist (hook/standalone/emotion/value/payoff/share)
# Stage 6  ensemble score + diversity (MMR) so you don't get 8 clips of the same point
def _rx(L):
    return [re.compile(p) for p in L]

HOOK_RX = _rx([r"\bwhat if\b", r"\bnobody (tells|talks|knows|told|wants)", r"\bthe (secret|truth|real reason|one thing)\b",
               r"\bmost people\b", r"\bhere'?s (the thing|why|how|what|the)", r"\byou (won'?t believe|need to|have to|must|never)\b",
               r"\b(biggest|worst|best) (mistake|lesson|thing|advice)\b", r"\bi was wrong\b", r"\bstop (doing|using|saying)\b",
               r"\bthe problem (is|with)\b", r"\bturns out\b", r"\blet me (tell|show|explain)\b", r"\bimagine\b",
               r"\bdid you know\b", r"\bthe reason\b", r"\bthis (changed|saved|ruined|blew)\b",
               r"\b(\d+|one|two|three|five|seven|ten) (ways|things|reasons|rules|steps|tips|mistakes|lessons)\b",
               r"\bthat'?s (crazy|insane|wild)\b", r"\bwhy (do|does|is|are|did|would)\b"])
STORY_RX = _rx([r"\bone (day|time)\b", r"\bwhen i was\b", r"\bi remember\b", r"\bso i (said|went|told|decided|started)\b",
                r"\band then\b", r"\bthat'?s when\b", r"\bnext thing\b", r"\bout of nowhere\b", r"\bi (couldn'?t|didn'?t) believe\b"])
PAYOFF_RX = _rx([r"\bthat'?s (why|how|what)\b", r"\bthe (lesson|takeaway|point|moral)\b", r"\bturns out\b", r"\bin the end\b",
                 r"\bso (the|if you|remember)\b", r"\bbottom line\b", r"\bthat'?s the (whole|real|only)\b", r"\bthe answer (is|was)\b",
                 r"\bwhich (means|is why)\b", r"\bhere'?s the (thing|kicker)\b"])
FILLER_RX = re.compile(r"\b(you know|i mean|kind of|sort of|basically|literally|i guess)\b")
FILLER = {"um", "uh", "uhm", "er", "ah", "hmm", "like"}
DEPEND_START = {"and", "so", "but", "because", "which", "that", "this", "it", "they", "he", "she", "then", "also", "or",
                "those", "these", "such", "anyway", "yeah", "right", "well", "cause", "plus", "however"}
EMO = set("""love hate afraid scared fear angry mad furious amazing incredible terrible horrible awful crazy insane wild
shocked shocking hilarious funny laugh cry cried tears pain hurt beautiful disgusting proud ashamed embarrassed
nightmare dream impossible unbelievable stupid genius brilliant brutal heartbreaking exciting thrilled obsessed
lied lie cheated betrayed destroyed saved won lost failed died""".split())


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def build_sentences(segs):
    sents, cur = [], []

    def flush():
        nonlocal cur
        if cur:
            sents.append({"text": " ".join(w["w"] for w in cur), "start": cur[0]["s"],
                          "end": cur[-1]["e"], "words": cur})
            cur = []
    for s in segs:
        for w in s["words"]:
            if cur and (w["s"] - cur[-1]["e"] > 1.0 or len(cur) >= 40):
                flush()
            cur.append(w)
            if re.search(r"[.!?…][\"')\]]*$", w["w"]):
                flush()
    flush()
    return sents


def hook_score(text):
    low = text.lower()
    toks = re.findall(r"[a-z']+", low)
    if not toks:
        return -2.0
    s = min(sum(1 for r in HOOK_RX if r.search(low)), 3) * 1.0
    s += 1.0 if "?" in text else 0
    s += 0.5 if re.search(r"\b(you|your)\b", low) else 0
    s += 0.5 if re.search(r"\d", low) else 0
    s += 0.7 if re.search(r"\b(never|always|nobody|everyone|worst|best|only|biggest|every)\b", low) else 0
    if len(toks) < 4:
        s -= 1.5
    if len(toks) > 35:
        s -= 0.7
    if toks[0] in DEPEND_START:
        s -= 1.5
    return s


def score_window(sents, i, j, ctx):
    S = sents[i:j + 1]
    text = " ".join(s["text"] for s in S)
    low = text.lower()
    start, end = S[0]["start"], S[-1]["end"]
    dur = max(end - start, 1.0)
    toks = re.findall(r"[a-z']+", low)
    n = max(len(toks), 1)
    content = [t for t in toks if t not in STOP and len(t) > 3]
    hook = hook_score(S[0]["text"]) + 0.5 * (hook_score(S[1]["text"]) if len(S) > 1 else 0)
    emo = sum(t in EMO for t in toks) / n * 15 + text.count("!") * 0.4
    story = min(sum(1 for r in STORY_RX if r.search(low)), 3) * 0.5
    payoff = min(sum(1 for r in PAYOFF_RX if r.search(low[-250:])), 2) * 0.9
    info = len(set(content)) / max(len(content), 1) * 2 + min(len(re.findall(r"\b\d[\d,.%]*\b", text)), 4) * 0.25
    top = Counter(content).most_common(5)
    focus = sum(c for _, c in top) / max(len(content), 1) * 4          # stays on ONE topic
    pace = 1 - min(abs(len(toks) / dur - 2.9) / 2.9, 1)
    length = 1 - min(abs(dur - 36) / 36, 1)                             # sweet spot ~36s
    filler = (sum(t in FILLER for t in toks) + len(FILLER_RX.findall(low))) / n
    audio = window_mean(ctx["energy"], 1, start, end) * 1.5 + window_mean(ctx["peaks"], 1, start, end) * 2.0
    mot = window_mean(ctx["motion"], ctx["mstep"], start, end) * 0.8
    end_ok = 0.4 if re.search(r"[.!]$", S[-1]["text"]) else -0.8
    total = 1.6 * hook + emo + story + payoff + info + focus + pace + length + audio + mot - filler * 8 + end_ok
    return total, {"hook": round(hook, 1), "emotion": round(emo, 1), "payoff": round(payoff, 1),
                   "focus": round(focus, 1), "audio": round(audio, 1)}


def _overlap(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(min(a1 - a0, b1 - b0), 1e-6)


def refine(sents, i, j, o, ctx):
    """Try trimming weak leading/trailing sentences or pulling in context; keep the best window."""
    best, best_s, best_b = (i, j), -1e9, {}
    for di in range(-2, 4):
        for dj in range(0, 3):
            ii, jj = i + di, j - dj
            if ii < 0 or jj >= len(sents) or ii > jj:
                continue
            dur = sents[jj]["end"] - sents[ii]["start"]
            if dur > o.max_len or dur < o.min_len * 0.8:
                continue
            sc, br = score_window(sents, ii, jj, ctx)
            sc -= 0.3 * abs(di)           # small penalty for moving far from original
            if sc > best_s:
                best, best_s, best_b = (ii, jj), sc, br
    return best[0], best[1], best_s, best_b


def llm_discover(sents, o, progress):
    """LLM reads the entire transcript in ~7 min chunks (with overlap) and nominates clip-worthy moments."""
    found, i = [], 0
    while i < len(sents):
        t0, j = sents[i]["start"], i
        while j < len(sents) and sents[j]["end"] - t0 < 420:
            j += 1
        lines = "\n".join(f"[{k}] {mmss(sents[k]['start'])} {sents[k]['text'][:200]}" for k in range(i, j))
        progress(f"AI reading transcript {mmss(t0)}–{mmss(sents[j - 1]['end'])}...")
        r = llm("You are an expert short-form video editor (TikTok/Reels/Shorts). Below is a transcript with sentence "
                "indices. Pick up to 3 moments that would work as STANDALONE viral clips of "
                f"{o.min_len}-{o.max_len} seconds. A great clip: opens with a strong hook in its FIRST sentence "
                "(bold claim, question, surprising fact, conflict), needs no prior context, stays on one idea, "
                "and ends on a payoff, punchline or conclusion. Avoid intros, sponsor reads, small talk and "
                "rambling. Return JSON: {\"moments\":[{\"start\":<first sentence index>,\"end\":<last sentence "
                "index>,\"why\":\"short reason\"}]}\n\nTRANSCRIPT:\n" + lines, timeout=300)
        moments = r.get("moments", []) if isinstance(r, dict) else []
        for m in moments if isinstance(moments, list) else []:
            try:
                a, b = int(m["start"]), int(m["end"])
            except (KeyError, TypeError, ValueError):
                continue
            a, b = max(a, i), min(b, j - 1)
            if a > b:
                continue
            while b > a and sents[b]["end"] - sents[a]["start"] > o.max_len:
                b -= 1
            if sents[b]["end"] - sents[a]["start"] >= o.min_len * 0.8:
                found.append((a, b, str(m.get("why", ""))))
        if j >= len(sents):
            break
        k = j
        while k > i + 1 and sents[j - 1]["end"] - sents[k - 1]["start"] < 60:
            k -= 1
        i = max(k, i + 1)
    return found


RUBRIC_W = {"hook": 0.25, "standalone": 0.20, "emotion": 0.15, "value": 0.15, "payoff": 0.15, "shareability": 0.10}


def rubric(c, sents):
    ctx = " ".join(s["text"] for s in sents[max(0, c["i"] - 2):c["i"]])
    r = llm("You are a harsh short-form content judge. Rate this candidate clip from 1-10 on each dimension. Be "
            "critical: the average clip is 4/10, reserve 8+ for exceptional. Dimensions: hook (does the FIRST "
            "sentence stop the scroll?), standalone (understandable with zero prior context?), emotion (does it "
            "make you feel something?), value (insight, fact, entertainment), payoff (satisfying ending?), "
            "shareability (would someone send it to a friend?). Also write a catchy title (<70 chars), a "
            "1-2 sentence caption, 4 hashtags, and one-sentence reason. Return JSON: {\"hook\":n,\"standalone\":n,"
            "\"emotion\":n,\"value\":n,\"payoff\":n,\"shareability\":n,\"title\":\"\",\"description\":\"\","
            "\"hashtags\":[],\"reason\":\"\"}\n\n"
            f"CONTEXT BEFORE THE CLIP (not part of it): {ctx[:400] or '(start of video)'}\n\n"
            f"CLIP:\n{c['text'][:1800]}")
    if not isinstance(r, dict):
        return None
    try:
        dims = {k: max(1.0, min(10.0, float(r[k]))) for k in RUBRIC_W}
    except (KeyError, TypeError, ValueError):
        return None
    r["dims"] = dims
    r["score"] = sum(dims[k] * w for k, w in RUBRIC_W.items()) * 10     # 10-100
    return r


def meta_heuristic(text):
    toks = [t for t in re.findall(r"[a-zA-Z']{4,}", text.lower()) if t not in STOP]
    sents = re.split(r"(?<=[.!?])\s", text.strip()) if text.strip() else ["Clip"]
    title = sents[0][:70] + ("…" if len(sents[0]) > 70 else "")
    desc = " ".join(sents[:2])[:220]
    tags = ["#" + w for w, _ in Counter(toks).most_common(4)] + ["#shorts", "#fyp"]
    return title, desc, tags


def _kw(text):
    return {w for w, _ in Counter(t for t in re.findall(r"[a-z']{4,}", text.lower()) if t not in STOP).most_common(8)}


def ollama_ready(progress):
    global OLLAMA_MODEL
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=3) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        progress("Ollama not running -> smart heuristic mode (install Ollama + `ollama pull llama3.1:8b` for far better picks)")
        return False
    if not models:
        progress("Ollama has no models -> run `ollama pull llama3.1:8b`. Using heuristic mode.")
        return False
    if OLLAMA_MODEL not in models and f"{OLLAMA_MODEL}:latest" not in models:
        OLLAMA_MODEL = models[0]
        progress(f"Using installed Ollama model: {OLLAMA_MODEL}")
    return True


def select_highlights(segs, o, energy, motion, total_dur, use_llm, progress=print):
    import bisect
    sents = build_sentences(segs)
    if not sents:
        return []
    mot, mstep = motion
    z = (energy - energy.mean()) / (energy.std() + 1e-6)
    ctx = {"energy": energy, "peaks": (z > 1.0).astype(float), "motion": mot, "mstep": mstep}

    # --- stage 2: enumerate + heuristic scoring of every window
    wins = []
    for i in range(len(sents)):
        if len(sents[i]["words"]) < 4:
            continue
        for j in range(i, len(sents)):
            dur = sents[j]["end"] - sents[i]["start"]
            if dur > o.max_len:
                break
            if dur >= o.min_len:
                wins.append((i, j))
    if not wins:
        i = 0
        while i < len(sents):
            j = i
            while j + 1 < len(sents) and sents[j + 1]["end"] - sents[i]["start"] <= o.max_len:
                j += 1
            wins.append((i, j))
            i = j + 1
    scored = sorted(((score_window(sents, i, j, ctx)[0], i, j) for i, j in wins), reverse=True)
    allraw = sorted(s for s, _, _ in scored)

    def pct(raw):
        return bisect.bisect_left(allraw, raw) / max(len(allraw), 1)

    def to_cand(i, j, raw, br, llm_pick=False, why=""):
        S = sents[i:j + 1]
        return {"i": i, "j": j, "start": S[0]["start"], "end": S[-1]["end"], "text": " ".join(s["text"] for s in S),
                "raw": raw, "pct": pct(raw), "br": br, "llm_pick": llm_pick, "why": why}

    pool, kept = [], []
    for raw, i, j in scored:
        a, b = sents[i]["start"], sents[j]["end"]
        if all(_overlap(a, b, k0, k1) <= 0.4 for k0, k1 in kept):
            kept.append((a, b))
            ii, jj, r2, br = refine(sents, i, j, o, ctx)
            pool.append(to_cand(ii, jj, r2, br))
        if len(pool) >= max(o.n * 3, 12):
            break

    # --- stage 3: LLM nominations from the full transcript
    if use_llm:
        for a, b, why in llm_discover(sents, o, progress):
            ii, jj, r2, br = refine(sents, a, b, o, ctx)
            pool.append(to_cand(ii, jj, r2, br, True, why))

    # dedupe overlapping candidates (prefer LLM-picked, then higher score)
    pool.sort(key=lambda c: (c["llm_pick"], c["raw"]), reverse=True)
    uniq = []
    for c in pool:
        if all(_overlap(c["start"], c["end"], u["start"], u["end"]) <= 0.6 for u in uniq):
            uniq.append(c)
    uniq.sort(key=lambda c: (c["llm_pick"], c["raw"]), reverse=True)
    uniq = uniq[:min(max(o.n * 2, 10), 20)] if use_llm else uniq[:max(o.n * 3, 12)]

    # --- stage 5: rubric rating + metadata
    for k, c in enumerate(uniq, 1):
        c["heur"] = 40 + 58 * c["pct"]
        title, desc, tags = meta_heuristic(c["text"])
        c.update(title=title, description=desc, hashtags=tags, scores={}, reason=c["why"] or
                 "Strong hook, energy and pacing (heuristic).")
        c["final"] = c["heur"]
        if use_llm:
            progress(f"AI judging finalist {k}/{len(uniq)}...")
            r = rubric(c, sents)
            if r:
                c["final"] = 0.65 * r["score"] + 0.35 * c["heur"] + (4 if c["llm_pick"] else 0)
                c["title"] = str(r.get("title") or title)
                c["description"] = str(r.get("description") or desc)
                if isinstance(r.get("hashtags"), list) and r["hashtags"]:
                    c["hashtags"] = ["#" + str(t).lstrip("#") for t in r["hashtags"]][:6] + ["#shorts"]
                c["reason"] = str(r.get("reason") or c["reason"])
                c["scores"] = {k2: round(v) for k2, v in r["dims"].items()}
            elif c["llm_pick"]:
                c["final"] += 4
        c["final"] = min(c["final"], 99)

    # --- stage 6: greedy MMR selection (quality minus similarity to what's already chosen)
    chosen, rest = [], list(uniq)
    while rest and len(chosen) < o.n:
        def mmr(c):
            sim = max((len(_kw(c["text"]) & _kw(s["text"])) / max(len(_kw(c["text"]) | _kw(s["text"])), 1)
                       for s in chosen), default=0)
            return c["final"] - 25 * sim
        best = max(rest, key=mmr)
        rest.remove(best)
        if all(_overlap(best["start"], best["end"], s["start"], s["end"]) <= 0.1 for s in chosen):
            chosen.append(best)
    out = []
    for c in sorted(chosen, key=lambda c: c["start"]):
        c["start"] = max(0.0, c["start"] - 0.2)                  # breathing room around speech
        c["end"] = min(total_dur, c["end"] + 0.4)
        c["virality"] = int(c["final"])
        out.append(c)
    return out


def broll_suggestions(words, start):
    seen, out = Counter(), []
    toks = [re.sub(r"\W", "", w["w"].lower()) for w in words]
    for t in toks:
        if len(t) >= 5 and t not in STOP:
            seen[t] += 1
    for kw, _ in seen.most_common(3):
        w = next(w for w, t in zip(words, toks) if t == kw)
        out.append({"keyword": kw, "time": round(w["s"] - start, 1),
                    "search": f"https://www.pexels.com/search/videos/{kw}/"})
    return out


# -------------------------------------------------- translation / dubbing
def translate_all(texts, src, to):
    if src == to:
        return texts
    try:  # free offline translation
        import argostranslate.package as ap, argostranslate.translate as at
        if not ({src, to} <= {l.code for l in at.get_installed_languages()}):
            ap.update_package_index()
            pk = next(p for p in ap.get_available_packages() if p.from_code == src and p.to_code == to)
            ap.install_from_path(pk.download())
        return [at.translate(t, src, to) for t in texts]
    except Exception:
        pass
    out = []
    for t in texts:
        j = llm(f'Translate to language code "{to}". Return JSON {{"translation":"..."}}. Text: {t}')
        if not j:
            raise RuntimeError("Translation failed. Install argostranslate (pip) or run Ollama.")
        out.append(str(j.get("translation", t)))
    return out


def pseudo_words(inside, texts, clip):
    words = []
    for s, t in zip(inside, texts):
        a, b = max(s["start"], clip["start"]), min(s["end"], clip["end"])
        ws = t.split()
        if not ws or b <= a:
            continue
        step = (b - a) / len(ws)
        words += [{"w": w, "s": a + i * step, "e": a + (i + 1) * step} for i, w in enumerate(ws)]
    return words


def build_dub(video, clip, inside, texts, o, tmp):
    import edge_tts
    voice = o.voice or VOICES.get(o.lang)
    if not voice:
        raise RuntimeError(f"No default voice for '{o.lang}'. Pass --voice (see: edge-tts --list-voices).")
    dur = clip["end"] - clip["start"]
    cmd = ["ffmpeg", "-y", "-ss", str(clip["start"]), "-t", str(dur), "-i", str(Path(video).resolve())]
    filt, labels = ["[0:a]volume=0.10[o]"], ["[o]"]
    for k, (s, t) in enumerate(zip(inside, texts), 1):
        p = tmp / f"tts{k}.mp3"
        asyncio.run(edge_tts.Communicate(t, voice).save(str(p)))
        slot = max(min(s["end"], clip["end"]) - max(s["start"], clip["start"]), 0.5)
        ratio = min(max(duration_of(p) / slot, 1.0), 1.8)
        delay = int(max(s["start"] - clip["start"], 0) * 1000)
        cmd += ["-i", str(p)]
        filt.append(f"[{k}:a]atempo={ratio:.3f},adelay={delay}|{delay}[t{k}]")
        labels.append(f"[t{k}]")
    filt.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=first[a]")
    out = tmp / "dub.m4a"
    run(cmd + ["-filter_complex", ";".join(filt), "-map", "[a]", "-t", str(dur), str(out)])
    return out


# --------------------------------------------------------------- captions
def ts(t):
    t = max(t, 0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def make_ass(words, clip_start, path, o, tw, th):
    st = STYLES.get(o.style, STYLES["yellow"])
    scale = tw / 1080
    align, mv = {"bottom": (2, int(th * 0.27)), "middle": (5, 0), "top": (8, int(th * 0.12))}[o.position]
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {tw}
PlayResY: {th}

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,{int(st['size'] * scale)},{st['p']},{st['s']},&H00000000,&H80000000,1,0,0,0,100,100,0,0,{st['bs']},{6 if st['bs'] == 1 else 12},2,{align},60,60,{mv},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines, n = [], max(o.words_per_card, 1)
    for k in range(0, len(words), n):
        grp = words[k:k + n]
        txt = ""
        for w in grp:
            word = w["w"].upper() if st["upper"] else w["w"]
            txt += (f"{{\\k{max(int((w['e'] - w['s']) * 100), 1)}}}" if st["karaoke"] else "") + word + " "
        lines.append(f"Dialogue: 0,{ts(grp[0]['s'] - clip_start)},{ts(grp[-1]['e'] - clip_start)},"
                     f"Default,,0,0,0,,{txt.strip()}")
    Path(path).write_text(head + "\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------- reframe
def track_x(video, start, dur, vw, cw, fps):
    import cv2, numpy as np
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    ts_, xs = [], []
    k = 0
    while k * 0.5 < dur:
        t = k * 0.5
        k += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + t) * 1000)
        ok, f = cap.read()
        if not ok:
            break
        sc = 480 / f.shape[1]
        g = cv2.cvtColor(cv2.resize(f, None, fx=sc, fy=sc), cv2.COLOR_BGR2GRAY)
        fc = det.detectMultiScale(g, 1.1, 5, minSize=(30, 30))
        ts_.append(t)
        if len(fc):
            b = max(fc, key=lambda b: b[2] * b[3])
            xs.append((b[0] + b[2] / 2) / sc)
        else:
            xs.append(None)
    cap.release()
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    last, filled = vals[0], []
    for x in xs:
        last = x if x is not None else last
        filled.append(last)
    cur, sm = filled[0], []
    for x in filled:
        cur += 0.3 * (x - cur)           # smooth camera movement
        sm.append(cur)
    frame_t = np.arange(int(dur * fps)) / fps
    cx = np.interp(frame_t, ts_, sm)
    return np.clip(cx - cw / 2, 0, vw - cw).astype(int)


def pick_broll(words, bdir, start, dur):
    files = [f for f in Path(bdir).glob("*") if f.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm")]
    ins, last = [], -99
    for w in words:
        wl, rel = re.sub(r"\W", "", w["w"].lower()), w["s"] - start
        if not wl or len(ins) >= 3 or rel < 1.5 or rel > dur - 3 or rel - last < 8:
            continue
        for f in files:
            if wl in re.split(r"[_\-\s.]+", f.stem.lower()):
                ins.append((f, rel))
                last = rel
                break
    return ins


# ------------------------------------------------------------------ render
def render_clip(video, clip, segs, src_lang, o, out_file, info):
    vw, vh, fps = info["w"], info["h"], info["fps"]
    tw, th = ASPECTS[o.aspect]
    cw = min(vw, int(vh * tw / th)) // 2 * 2
    ch = int(cw * th / tw) // 2 * 2
    if ch > vh:
        ch = vh // 2 * 2
        cw = int(ch * tw / th) // 2 * 2
    start, dur = clip["start"], clip["end"] - clip["start"]
    video = Path(video).resolve()
    tmp = Path(tempfile.mkdtemp())
    try:
        inside = [s for s in segs if s["end"] > start and s["start"] < clip["end"]]
        words = [w for s in inside for w in s["words"] if start <= w["s"] < clip["end"]]
        dub = None
        if o.lang and o.lang != src_lang and inside:
            texts = translate_all([s["text"] for s in inside], src_lang, o.lang)
            words = pseudo_words(inside, texts, clip)
            if o.dub:
                dub = build_dub(video, clip, inside, texts, o, tmp)
        brolls = pick_broll(words, o.broll_dir, start, dur) if o.broll_dir else []

        try:
            import cv2
            pipe = o.track and (cw, ch) != (vw, vh)
        except ImportError:
            pipe = False

        cmd, parts = ["ffmpeg", "-y"], []
        if pipe:
            cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{tw}x{th}", "-r", str(fps), "-i", "-"]
            cur = "[0:v]"
            if dub:
                cmd += ["-i", str(dub)]
            else:
                cmd += ["-ss", str(start), "-t", str(dur), "-i", str(video)]
            aidx, nxt = 1, 2
        else:
            cmd += ["-ss", str(start), "-t", str(dur), "-i", str(video)]
            parts.append(f"[0:v]crop={cw}:{ch}:(in_w-{cw})/2:(in_h-{ch})/2,scale={tw}:{th}[c0]")
            cur = "[c0]"
            if dub:
                cmd += ["-i", str(dub)]
                aidx, nxt = 1, 2
            else:
                aidx, nxt = 0, 1
        for i, (f, a) in enumerate(brolls):
            k = nxt + i
            cmd += ["-t", "3", "-i", str(f.resolve())]
            parts.append(f"[{k}:v]scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},"
                         f"setpts=PTS-STARTPTS+{a:.2f}/TB[b{i}]")
            parts.append(f"{cur}[b{i}]overlay=enable='between(t,{a:.2f},{a + 3:.2f})':eof_action=pass[v{i}]")
            cur = f"[v{i}]"
        if o.captions and words:
            make_ass(words, start, tmp / "subs.ass", o, tw, th)
            parts.append(f"{cur}subtitles=subs.ass[vout]")
            cur = "[vout]"
        if parts:
            cmd += ["-filter_complex", ";".join(parts)]
        cmd += ["-map", cur if parts else "0:v", "-map", f"{aidx}:a?", "-t", str(dur),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(Path(out_file).resolve())]

        if not pipe:
            run(cmd, cwd=tmp)
        else:
            import numpy as np
            n = int(dur * fps)
            xs = track_x(video, start, dur, vw, cw, fps)
            if xs is None:
                xs = np.full(n, (vw - cw) // 2)
            y = (vh - ch) // 2
            log = open(tmp / "ff.log", "wb")
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log, cwd=tmp)
            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            try:
                for i in range(n):
                    ok, fr = cap.read()
                    if not ok:
                        break
                    x = int(xs[min(i, len(xs) - 1)])
                    fr = cv2.resize(fr[y:y + ch, x:x + cw], (tw, th), interpolation=cv2.INTER_AREA)
                    proc.stdin.write(fr.tobytes())
            except BrokenPipeError:
                pass
            finally:
                cap.release()
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.wait()
                log.close()
            if proc.returncode != 0:
                raise RuntimeError("ffmpeg failed:\n" + (tmp / "ff.log").read_text(errors="ignore")[-1500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- pipeline
def save_project(job, data):
    (job / "project.json").write_text(json.dumps(data, indent=1))
    with open(job / "posts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "title", "description", "hashtags", "virality"])
        for c in data["clips"]:
            w.writerow([Path(c["file"]).name, c["title"], c["description"], " ".join(c["hashtags"]), c["virality"]])


def process(source, o, progress=print):
    global OLLAMA_MODEL
    if o.llm_model:
        OLLAMA_MODEL = o.llm_model
    need("ffmpeg"); need("ffprobe")
    job = OUT_ROOT / uuid.uuid4().hex[:8]
    job.mkdir()
    if re.match(r"https?://", source):
        progress("Downloading video...")
        video = download(source, job)
    else:
        video = Path(source)
        if not video.exists():
            raise RuntimeError("File not found")
    info = probe(video)
    segs, lang = transcribe(video, o, progress)
    if not segs:
        raise RuntimeError("No speech detected in this video.")
    progress("Analyzing audio and visuals...")
    try:
        import numpy as np
        energy = audio_energy(video)
        motion = visual_motion(video)
    except ImportError:
        import numpy as np
        energy, motion = np.zeros(1), (np.zeros(1), 2.0)
    use_llm = o.use_llm and ollama_ready(progress)
    progress("Finding best moments...")
    picks = select_highlights(segs, o, energy, motion, info["dur"], use_llm, progress)
    if not picks:
        raise RuntimeError("Could not find any clip-worthy moments. Try lowering --min.")
    clips = []
    for idx, c in enumerate(picks, 1):
        progress(f"Rendering clip {idx}/{len(picks)}...")
        f = job / f"clip_{idx:02d}.mp4"
        render_clip(video, c, segs, lang, o, f, info)
        words = [w for s in segs for w in s["words"] if c["start"] <= w["s"] < c["end"]]
        clips.append({"file": str(f), "start": round(c["start"], 1), "end": round(c["end"], 1),
                      "virality": c["virality"], "reason": c["reason"], "title": c["title"],
                      "description": c["description"], "hashtags": c["hashtags"], "scores": c.get("scores", {}),
                      "broll_suggestions": broll_suggestions(words, c["start"])})
    save_project(job, {"id": job.name, "video": str(video), "info": info, "lang": lang,
                       "opts": asdict(o), "segments": segs, "clips": clips})
    progress(f"Done! {len(clips)} clips in {job}")
    return job, clips


def rerender(job_id, idx, start, end, overrides, progress=print):
    job = OUT_ROOT / job_id
    p = json.loads((job / "project.json").read_text())
    o = Opts(**{**p["opts"], **overrides})
    c = p["clips"][idx]
    c["start"], c["end"] = float(start), float(end)
    progress("Re-rendering clip...")
    render_clip(p["video"], c, p["segments"], p["lang"], o, c["file"], p["info"])
    p["clips"][idx] = c
    save_project(job, p)
    return c


def upload_youtube(file, title, desc, tags, privacy="private"):
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    tok = Path("youtube_token.json")
    creds = Credentials.from_authorized_user_file(str(tok), scopes) if tok.exists() else None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = InstalledAppFlow.from_client_secrets_file("client_secret.json", scopes).run_local_server(port=0)
        tok.write_text(creds.to_json())
    yt = build("youtube", "v3", credentials=creds)
    body = {"snippet": {"title": title[:100], "description": desc + "\n\n" + " ".join(tags) + " #Shorts",
                        "tags": [t.lstrip("#") for t in tags], "categoryId": "22"},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
    req = yt.videos().insert(part="snippet,status", body=body, media_body=MediaFileUpload(file, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    return "https://youtube.com/shorts/" + resp["id"]


def watch_channel(url, latest, interval, o):
    seen_f = OUT_ROOT / "seen.json"
    while True:
        seen = set(json.loads(seen_f.read_text())) if seen_f.exists() else set()
        for v in list_channel(url, latest):
            if v not in seen:
                print("New video:", v)
                try:
                    process(v, o)
                    seen.add(v)
                    seen_f.write_text(json.dumps(sorted(seen)))
                except Exception as e:
                    print("Failed:", e)
        if not interval:
            break
        print(f"Sleeping {interval}s...")
        time.sleep(interval)


# ------------------------------------------------------------------ web UI
PAGE = r"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>ClipForge</title>
<style>body{font-family:system-ui;max-width:980px;margin:24px auto;padding:0 14px;background:#0f1115;color:#eee}
input,select,button,textarea{padding:9px;border-radius:8px;border:1px solid #333;background:#1b1e26;color:#eee;margin:3px 0}
button{background:#6c5ce7;border:0;cursor:pointer;font-weight:600}button.s{background:#2d3140}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.card{background:#1b1e26;border-radius:12px;padding:12px;margin:10px 0}
video{width:210px;border-radius:8px}.clips{display:flex;flex-direction:column}.clip{display:flex;gap:14px;flex-wrap:wrap}
small{color:#9aa}.v{background:#00b894;color:#000;border-radius:6px;padding:2px 8px;font-weight:700}a{color:#a29bfe}</style>
<h1>🎬 ClipForge</h1><small>Unlimited local clip generation · no credits · no watermark</small>
<div class=card><form id=f enctype=multipart/form-data>
<input name=url placeholder="Video URL (YouTube etc.)" style="width:100%"><br>
<small>or upload</small> <input type=file name=file accept="video/*"><br>
<div class=row>
<label>Clips <input name=n type=number value=6 min=1 max=30 style="width:62px"></label>
<label>Min s <input name=min type=number value=20 style="width:62px"></label>
<label>Max s <input name=max type=number value=60 style="width:62px"></label>
<select name=aspect><option>9:16</option><option>1:1</option><option>4:5</option><option>16:9</option></select>
<select name=style><option>yellow</option><option>green</option><option>pink</option><option>minimal</option><option>boxed</option></select>
<select name=position><option>bottom</option><option>middle</option><option>top</option></select>
<select name=model><option selected>tiny</option><option>small</option><option>medium</option></select></div>
<div class=row><label><input type=checkbox name=captions checked> captions</label>
<label><input type=checkbox name=track checked> face tracking</label>
<label><input type=checkbox name=use_llm checked> AI scoring (Ollama)</label>
<input name=lang placeholder="translate to (es, fr, tl…)" style="width:170px">
<label><input type=checkbox name=dub> AI dub</label>
<input name=broll_dir placeholder="B-roll folder (optional)" style="width:190px">
<input name=llm_model placeholder="Ollama model (e.g. llama3.1:8b)" style="width:210px"></div>
<button>Generate clips</button></form></div>
<div id=status></div><h3>Projects</h3><div id=projects></div><div id=out class=clips></div>
<script>
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function poll(id,done){const t=setInterval(async()=>{const s=await (await fetch('/status/'+id)).json();
status.textContent=s.error?'Error: '+s.error:s.msg;if(s.done||s.error){clearInterval(t);if(s.done)done(s)}},1500)}
function show(job,clips){out.innerHTML='';clips.forEach((c,i)=>{const u='/file/'+job+'/'+c.file.split(/[\\/]/).pop()+'?t='+Date.now();
out.innerHTML+=`<div class=card><div class=clip><video src="${u}" controls></video><div style="flex:1;min-width:240px">
<span class=v>🔥 ${c.virality}</span> <b>${esc(c.title)}</b><br><small>${esc(c.reason)}</small><br><small>${c.scores?Object.entries(c.scores).map(([k,v])=>k+' '+v).join(' · '):''}</small><p>${esc(c.description)}</p>
<div>${c.hashtags.map(esc).join(' ')}</div>
<small>B-roll ideas: ${(c.broll_suggestions||[]).map(b=>`<a target=_blank href="${b.search}">${esc(b.keyword)}@${b.time}s</a>`).join(', ')||'-'}</small><br>
<div class=row><input id=s${i} value=${c.start} style="width:70px"><input id=e${i} value=${c.end} style="width:70px">
<button class=s onclick="rr('${job}',${i})">Trim &amp; re-render</button>
<a href="${u}" download>Download</a>
<button class=s onclick="yt('${job}',${i})">Upload to YouTube</button></div></div></div></div>`})}
f.onsubmit=async e=>{e.preventDefault();out.innerHTML='';
const {id}=await (await fetch('/start',{method:'POST',body:new FormData(f)})).json();
poll(id,s=>{show(s.job,s.clips);loadProjects()})}
async function rr(job,i){const b=new FormData();b.append('job',job);b.append('idx',i);b.append('start',document.getElementById('s'+i).value);b.append('end',document.getElementById('e'+i).value);
const {id}=await (await fetch('/rerender',{method:'POST',body:b})).json();poll(id,async()=>open_(job))}
async function yt(job,i){const b=new FormData();b.append('job',job);b.append('idx',i);
const {id}=await (await fetch('/youtube',{method:'POST',body:b})).json();poll(id,s=>status.innerHTML='Uploaded: <a target=_blank href="'+s.msg+'">'+s.msg+'</a>')}
async function open_(job){const p=await (await fetch('/project/'+job)).json();show(job,p.clips)}
async function loadProjects(){const l=await (await fetch('/projects')).json();
projects.innerHTML=l.map(p=>`<span class=card style="display:inline-block;margin-right:8px"><a href="#" onclick="open_('${p.id}');return false">${p.id}</a> · ${p.clips} clips
<button class=s onclick="fetch('/project/${p.id}',{method:'DELETE'}).then(loadProjects)">✕</button></span>`).join('')||'<small>none yet</small>'}
loadProjects()</script>"""


def create_app():
    from flask import Flask, request, jsonify, send_from_directory, Response
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("CLIPFORGE_MAX_UPLOAD_MB", "500")) * 1024 * 1024
    jobs = {}
    job_lock = threading.Lock()

    # A Render-hosted instance must never start without its access credentials.
    auth_user = os.getenv("CLIPFORGE_AUTH_USERNAME", "")
    auth_password = os.getenv("CLIPFORGE_AUTH_PASSWORD", "")
    if os.getenv("RENDER") and (not auth_user or not auth_password):
        raise RuntimeError("Set CLIPFORGE_AUTH_USERNAME and CLIPFORGE_AUTH_PASSWORD in the service environment.")

    @app.before_request
    def require_login():
        if request.path == "/healthz":
            return None
        # Local desktop use remains convenient; Render is fail-closed above.
        if not auth_user or not auth_password:
            return None
        credentials = request.authorization
        supplied_user = credentials.username if credentials else ""
        supplied_password = credentials.password if credentials else ""
        valid = hmac.compare_digest(supplied_user, auth_user) and hmac.compare_digest(supplied_password, auth_password)
        if not valid:
            return Response("Sign in required", 401, {"WWW-Authenticate": 'Basic realm="ClipForge", charset="UTF-8"'})
        return None

    @app.get("/healthz")
    def healthz():
        return jsonify(ok=True)

    def spawn(fn):
        if not job_lock.acquire(blocking=False):
            return jsonify(error="ClipForge is already processing a job. Please wait for it to finish."), 409
        jid = uuid.uuid4().hex
        jobs[jid] = {"msg": "Queued...", "done": False}

        def work():
            try:
                jobs[jid].update(fn(lambda m: jobs[jid].update(msg=m)), done=True)
            except Exception as ex:
                jobs[jid].update(error=str(ex))
            finally:
                job_lock.release()
        threading.Thread(target=work, daemon=True).start()
        return jsonify(id=jid)

    @app.get("/")
    def index():
        return PAGE

    @app.post("/start")
    def start():
        fm = request.form
        src = fm.get("url", "").strip()
        up = request.files.get("file")
        if up and up.filename:
            p = OUT_ROOT / f"upload_{uuid.uuid4().hex[:6]}_{Path(up.filename).name}"
            up.save(p)
            src = str(p)
        if not src:
            return jsonify(error="Add a video URL or upload a video file."), 400
        o = Opts(n=int(fm["n"]), min_len=int(fm["min"]), max_len=int(fm["max"]), model=fm["model"],
                 captions="captions" in fm, style=fm["style"], position=fm["position"], aspect=fm["aspect"],
                 track="track" in fm, use_llm="use_llm" in fm, lang=fm.get("lang", "").strip(),
                 dub="dub" in fm, broll_dir=fm.get("broll_dir", "").strip(),
                 llm_model=fm.get("llm_model", "").strip())

        def job(progress):
            j, clips = process(src, o, progress)
            return {"job": j.name, "clips": clips, "msg": "Done!"}
        return spawn(job)

    @app.post("/rerender")
    def rr():
        fm = request.form

        def job(progress):
            rerender(fm["job"], int(fm["idx"]), fm["start"], fm["end"], {}, progress)
            return {"msg": "Done!"}
        return spawn(job)

    @app.post("/youtube")
    def yt():
        fm = request.form

        def job(progress):
            progress("Uploading to YouTube (requires your Google OAuth setup)...")
            c = json.loads((OUT_ROOT / fm["job"] / "project.json").read_text())["clips"][int(fm["idx"])]
            return {"msg": upload_youtube(c["file"], c["title"], c["description"], c["hashtags"])}
        return spawn(job)

    @app.get("/status/<jid>")
    def status(jid):
        return jsonify(jobs.get(jid, {"error": "unknown job"}))

    @app.get("/projects")
    def projects():
        out = []
        for d in sorted(OUT_ROOT.glob("*/project.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            out.append({"id": d.parent.name, "clips": len(json.loads(d.read_text())["clips"])})
        return jsonify(out)

    @app.get("/project/<pid>")
    def project(pid):
        if not re.fullmatch(r"[0-9a-f]{8}", pid):
            return jsonify(error="Invalid project id"), 404
        target = OUT_ROOT / pid / "project.json"
        if not target.is_file():
            return jsonify(error="Project not found"), 404
        p = json.loads(target.read_text())
        p.pop("segments", None)
        return jsonify(p)

    @app.delete("/project/<pid>")
    def delete(pid):
        if re.fullmatch(r"[0-9a-f]{8}", pid):
            shutil.rmtree(OUT_ROOT / pid, ignore_errors=True)
        return jsonify(ok=True)

    @app.get("/file/<path:p>")
    def file(p):
        return send_from_directory(OUT_ROOT.resolve(), p)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="Video is larger than the configured upload limit."), 413

    return app


# Expose a WSGI application for Gunicorn/Render.
app = create_app()


def serve():
    print("Open http://localhost:5000")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ClipForge v2 - local AI video clipper")
    ap.add_argument("source", nargs="?", help="video URL or local file")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--min", type=int, default=20)
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--model", default="small", help="tiny|base|small|medium|large-v3")
    ap.add_argument("--source-lang", default="")
    ap.add_argument("--style", default="yellow", choices=list(STYLES))
    ap.add_argument("--position", default="bottom", choices=["bottom", "middle", "top"])
    ap.add_argument("--aspect", default="9:16", choices=list(ASPECTS))
    ap.add_argument("--lang", default="", help="translate captions to this language code")
    ap.add_argument("--dub", action="store_true", help="also dub audio into --lang")
    ap.add_argument("--voice", default="")
    ap.add_argument("--broll-dir", default="")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--ai-model", default="", help="Ollama model, e.g. llama3.1:8b or qwen2.5:7b")
    ap.add_argument("--channel", help="channel/playlist URL to auto-import")
    ap.add_argument("--latest", type=int, default=3)
    ap.add_argument("--watch", type=int, default=0, help="re-check channel every N seconds")
    ap.add_argument("--upload-youtube", action="store_true")
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    opts = Opts(n=a.clips, min_len=a.min, max_len=a.max, model=a.model, source_lang=a.source_lang,
                captions=not a.no_captions, style=a.style, position=a.position, aspect=a.aspect,
                track=not a.no_track, use_llm=not a.no_llm, lang=a.lang, dub=a.dub, voice=a.voice,
                broll_dir=a.broll_dir, llm_model=a.ai_model)
    try:
        if a.serve:
            serve()
        elif a.channel:
            watch_channel(a.channel, a.latest, a.watch, opts)
        elif a.source:
            _, res = process(a.source, opts)
            if a.upload_youtube:
                for c in res:
                    print("Uploaded:", upload_youtube(c["file"], c["title"], c["description"], c["hashtags"]))
        else:
            ap.print_help()
    except RuntimeError as e:
        sys.exit(str(e))
