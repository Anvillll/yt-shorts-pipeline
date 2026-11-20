import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
import cv2
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from faster_whisper import WhisperModel
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from nudenet import NudeDetector

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

PROCESSED_FILE = Path("processed.txt")
PROCESSED = set(PROCESSED_FILE.read_text().splitlines()) if PROCESSED_FILE.exists() else set()

MAX_YT_PER_DAY = 1  # 1 YouTube Short per UTC day
MAX_IG_PER_DAY = 3  # Up to 3 Instagram Reels per UTC day (Morning, Noon, Night)
MAX_UPLOAD_ATTEMPTS_PER_RUN = 3  # Max candidate clips to render/attempt per single CI run
FALLBACK_HOURS = [24, 48, 72, 168, 720]  # Progressive fallback windows

PRIORITY_STREAMERS = [
    "kaicenat", "ishowspeed", "xqc", "caseoh_", "jynxzi",
    "adinross", "stableronaldo", "plaqueboymax"
]

BLOCKED_KEYWORDS = [
    "strip", "stripclub", "club", "nsfw", "nude", "nudity", "naked",
    "18+", "xxx", "porn", "hentai", "boobs", "booty", "lapdance",
    "unicorn", "vanilla"
]

EXPOSED_SENSITIVE_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


def mark_processed(clip_id):
    PROCESSED.add(clip_id)
    with PROCESSED_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{clip_id}\n")


def is_ascii_english_title(title):
    if not title:
        return False
    ascii_chars = sum(1 for c in title if ord(c) < 128)
    return (ascii_chars / len(title)) >= 0.80


def has_blocked_keywords(title):
    t_lower = title.lower()
    for word in BLOCKED_KEYWORDS:
        if re.search(rf"\b{re.escape(word)}\b", t_lower):
            return True
        if word in ["18+", "xxx"] and word in t_lower:
            return True
    return False


def contains_nsfw(video_path):
    temp_dir = tempfile.mkdtemp(prefix="nude_check_")
    try:
        frame_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "fps=1",
            "-q:v", "2",
            frame_pattern
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        frames = list(Path(temp_dir).glob("*.jpg"))
        if not frames:
            return False

        detector = NudeDetector()
        for frame in frames:
            detections = detector.detect(str(frame))
            for d in detections:
                label = d.get("class")
                score = d.get("score", 0.0)
                if label in EXPOSED_SENSITIVE_CLASSES and score >= 0.45:
                    print(f"NSFW detected in frame {frame.name}: {label} ({score:.2f})")
                    return True
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_video_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=True)
    return float(res.stdout.strip())


def detect_streamer_cam(video_path):
    """
    Sample 16 evenly spaced frames using OpenCV.
    Detect faces using BOTH frontalface and profileface Haar Cascade classifiers
    (including horizontally-flipped frames for right-side profiles).
    Filter for faces on sides (face_cx < 0.38 * width or face_cx > 0.62 * width).
    If >= 2 frames detect side face, compute median and return crop box with 1080:680
    aspect ratio. If < 2 faces detected, use smart corner fallback (top-left 32% crop).
    Returns None only if the video cannot be opened at all.
    """
    try:
        if not hasattr(cv2, "CascadeClassifier"):
            print("Warning: cv2.CascadeClassifier is not available. Falling back to full-screen vertical framing.")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or width <= 0 or height <= 0:
            cap.release()
            return None

        # Load both frontal and profile cascades
        haarcascades_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
        frontal_path = haarcascades_dir + "haarcascade_frontalface_default.xml"
        profile_path = haarcascades_dir + "haarcascade_profileface.xml"

        frontal_cascade = cv2.CascadeClassifier(frontal_path)
        profile_cascade = cv2.CascadeClassifier(profile_path)

        has_frontal = not frontal_cascade.empty()
        has_profile = not profile_cascade.empty()

        if not has_frontal and not has_profile:
            print(f"Warning: Could not load any Haar cascade. Falling back to corner crop.")
            cap.release()
            # Smart corner fallback
            crop_w = int(width * 0.32)
            crop_h = int(crop_w / (1080.0 / 680.0))
            crop_w = crop_w - (crop_w % 2)
            crop_h = crop_h - (crop_h % 2)
            print(f"Using smart corner fallback: top-left crop ({crop_w}x{crop_h})")
            return (0, 0, crop_w, crop_h)

        frame_indices = np.linspace(0, total_frames - 1, num=16, dtype=int)
        side_faces = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            all_detections = []

            # Frontal face detection
            if has_frontal:
                detected = frontal_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(36, 36)
                )
                if detected is not None and len(detected) > 0:
                    all_detections.extend(detected.tolist())

            # Profile face detection (normal orientation)
            if has_profile:
                detected_p = profile_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(36, 36)
                )
                if detected_p is not None and len(detected_p) > 0:
                    all_detections.extend(detected_p.tolist())

                # Profile detection on horizontally flipped frame (catches right-side profiles)
                flipped = cv2.flip(gray, 1)
                detected_pf = profile_cascade.detectMultiScale(
                    flipped, scaleFactor=1.1, minNeighbors=3, minSize=(36, 36)
                )
                if detected_pf is not None and len(detected_pf) > 0:
                    for (fx, fy, fw, fh) in detected_pf:
                        # Mirror x coordinate back to original frame
                        orig_x = width - fx - fw
                        all_detections.append([orig_x, fy, fw, fh])

            if all_detections:
                for (x, y, w, h) in all_detections:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    if cx < 0.38 * width or cx > 0.62 * width:
                        side_faces.append((cx, cy, float(w), float(h)))
                        break

        cap.release()

        if len(side_faces) < 2:
            # Smart corner fallback: top-left 32% crop at 1080:680 ratio
            crop_w = int(width * 0.32)
            crop_h = int(crop_w / (1080.0 / 680.0))
            crop_w = crop_w - (crop_w % 2)
            crop_h = crop_h - (crop_h % 2)
            print(f"Streamer cam detection: Only {len(side_faces)}/16 side faces detected. Using smart corner fallback ({crop_w}x{crop_h}).")
            return (0, 0, crop_w, crop_h)

        med_cx = float(np.median([f[0] for f in side_faces]))
        med_cy = float(np.median([f[1] for f in side_faces]))
        med_bw = float(np.median([f[2] for f in side_faces]))
        med_bh = float(np.median([f[3] for f in side_faces]))
        print(f"Streamer cam detected! Detected {len(side_faces)} side faces. Median center: ({med_cx:.1f}, {med_cy:.1f}), size: ({med_bw:.1f}, {med_bh:.1f})")

        target_aspect = 1080.0 / 680.0
        crop_h = int(med_bh * 3.2)
        crop_w = int(crop_h * target_aspect)

        min_h = int(height * 0.35)
        if crop_h < min_h:
            crop_h = min_h
            crop_w = int(crop_h * target_aspect)

        if crop_w > width:
            crop_w = width
            crop_h = int(crop_w / target_aspect)
        if crop_h > height:
            crop_h = height
            crop_w = int(crop_h * target_aspect)

        target_cy = med_cy + med_bh * 0.15
        crop_x = int(med_cx - crop_w / 2.0)
        crop_y = int(target_cy - crop_h / 2.0)

        crop_x = max(0, min(crop_x, width - crop_w))
        crop_y = max(0, min(crop_y, height - crop_h))

        crop_w = crop_w - (crop_w % 2)
        crop_h = crop_h - (crop_h % 2)
        crop_x = crop_x - (crop_x % 2)
        crop_y = crop_y - (crop_y % 2)

        return (crop_x, crop_y, crop_w, crop_h)

    except Exception as e:
        print(f"Warning: Streamer cam detection encountered an error ({e}). Falling back to full-screen vertical framing.")
        return None


def get_daily_counts():
    """
    Reads daily_count.txt supporting both:
    - 'YYYY-MM-DD:yt=X,ig=Y'
    - 'YYYY-MM-DD:X' (legacy format treated as yt=X, ig=0)
    Returns (yt_count, ig_count) for today's UTC date.
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count_file = Path("daily_count.txt")
    if not count_file.exists():
        return 0, 0

    content = count_file.read_text().strip()
    if ":" not in content:
        return 0, 0

    saved_date, counts_part = content.split(":", 1)
    if saved_date != today_str:
        return 0, 0

    yt_count = 0
    ig_count = 0

    if "yt=" in counts_part or "ig=" in counts_part:
        for item in counts_part.split(","):
            item = item.strip()
            if item.startswith("yt="):
                try:
                    yt_count = int(item.split("=")[1])
                except ValueError:
                    pass
            elif item.startswith("ig="):
                try:
                    ig_count = int(item.split("=")[1])
                except ValueError:
                    pass
    else:
        # Legacy single integer format
        try:
            yt_count = int(counts_part.strip())
        except ValueError:
            pass

    return yt_count, ig_count


def increment_daily_count(platform):
    """
    Increments 'yt' or 'ig' counter for today's UTC date.
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yt_count, ig_count = get_daily_counts()

    if platform == "yt":
        yt_count += 1
    elif platform == "ig":
        ig_count += 1

    count_file = Path("daily_count.txt")
    count_file.write_text(f"{today_str}:yt={yt_count},ig={ig_count}", encoding="utf-8")
    print(f"Updated daily count for {today_str}: yt={yt_count}, ig={ig_count}")


def get_twitch_token():
    r = requests.post("https://id.twitch.tv/oauth2/token", data={
        "client_id": os.environ["TWITCH_CLIENT_ID"],
        "client_secret": os.environ["TWITCH_CLIENT_SECRET"],
        "grant_type": "client_credentials",
    }, timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]


def get_gta6_game_id(token):
    headers = {"Client-ID": os.environ["TWITCH_CLIENT_ID"], "Authorization": f"Bearer {token}"}
    r = requests.get("https://api.twitch.tv/helix/games",
                     headers=headers, params={"name": "Grand Theft Auto VI"}, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    return data[0]["id"] if data else None


def get_broadcaster_ids(token, usernames):
    if not usernames:
        return []
    headers = {"Client-ID": os.environ["TWITCH_CLIENT_ID"], "Authorization": f"Bearer {token}"}
    params = [("login", u) for u in usernames]
    r = requests.get("https://api.twitch.tv/helix/users", headers=headers, params=params, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    return [u["id"] for u in data if "id" in u]


def fetch_clips_for_broadcaster(token, broadcaster_id, hours):
    headers = {"Client-ID": os.environ["TWITCH_CLIENT_ID"], "Authorization": f"Bearer {token}"}
    started_at = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
        "broadcaster_id": broadcaster_id,
        "started_at": started_at,
        "first": 20,
    }, timeout=15)
    r.raise_for_status()
    return r.json().get("data", [])


def fetch_clips_for_window(token, game_id, hours):
    headers = {"Client-ID": os.environ["TWITCH_CLIENT_ID"], "Authorization": f"Bearer {token}"}
    started_at = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
        "game_id": game_id,
        "started_at": started_at,
        "first": 50,
    }, timeout=15)
    r.raise_for_status()
    clips = sorted(r.json().get("data", []), key=lambda c: c.get("view_count", 0), reverse=True)
    return clips


def format_ass_time(sec):
    h, m = int(sec // 3600), int((sec % 3600) // 60)
    s, cs = int(sec % 60), int((sec % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def transcribe_and_verify(video_path):
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(video_path), word_timestamps=True)

    if info.language != "en" or info.language_probability < 0.75:
        print(f"Whisper rejected audio: detected language='{info.language}' with prob={info.language_probability:.2f}")
        return None, None, None

    all_words = []
    full_transcript_parts = []
    for seg in segments:
        full_transcript_parts.append(seg.text)
        for w in seg.words or []:
            all_words.append(w)

    transcript = " ".join(full_transcript_parts).strip()
    return all_words, info, transcript


def build_jumpcut_timeline(all_words, original_duration):
    """
    Groups consecutive words into active speech segments whenever silence gap <= 0.85s.
    Pads each active speech segment by 0.15s before start and 0.20s after end.
    Keeps total kept duration >= 10s. If trimmed < 10s, keeps original untrimmed duration.
    Ensures total kept duration is capped at 58.5s so video strictly qualifies as a Short/Reel.
    Remaps word timestamps to match jump-cut timeline.
    Returns: timeline_segments (list of dicts with start, end), remapped_words, total_duration
    """
    MAX_SHORT_DURATION = 58.5

    if not all_words:
        dur = min(original_duration, MAX_SHORT_DURATION)
        timeline = [{"start": 0.0, "end": dur}]
        return timeline, [], dur

    # 1. Group words where gap <= 0.85s
    raw_speech_groups = []
    current_group = [all_words[0]]

    for w in all_words[1:]:
        gap = w.start - current_group[-1].end
        if gap <= 0.85:
            current_group.append(w)
        else:
            raw_speech_groups.append(current_group)
            current_group = [w]
    if current_group:
        raw_speech_groups.append(current_group)

    # 2. Pad each segment: start - 0.15s, end + 0.20s, and clamp to [0, original_duration]
    padded_segments = []
    for grp in raw_speech_groups:
        start = max(0.0, grp[0].start - 0.15)
        end = min(original_duration, grp[-1].end + 0.20)
        padded_segments.append({"start": start, "end": end, "words": grp})

    # 3. Merge overlapping padded segments
    merged_speech_segments = []
    for seg in padded_segments:
        if not merged_speech_segments:
            merged_speech_segments.append(seg)
        else:
            prev = merged_speech_segments[-1]
            if seg["start"] <= prev["end"]:
                prev["end"] = max(prev["end"], seg["end"])
                prev["words"].extend(seg["words"])
            else:
                merged_speech_segments.append(seg)

    total_kept_speech = sum(s["end"] - s["start"] for s in merged_speech_segments)

    # 4. If total speech duration is < 10s, keep original untrimmed duration (capped at 58.5s)
    if total_kept_speech < 10.0:
        dur = min(original_duration, MAX_SHORT_DURATION)
        print(f"Total speech duration ({total_kept_speech:.2f}s) < 10s. Keeping original duration capped at {dur:.2f}s.")
        timeline = [{"start": 0.0, "end": dur}]
        remapped_words = [
            {"word": w.word, "start": w.start, "end": min(w.end, dur)}
            for w in all_words
            if w.start < dur
        ]
        return timeline, remapped_words, dur

    # Build jump-cut timeline (only kept speech segments are retained)
    timeline = []
    remapped_words = []
    accumulated_time = 0.0

    for seg in merged_speech_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_dur = seg_end - seg_start

        if accumulated_time + seg_dur > MAX_SHORT_DURATION:
            allowed_dur = MAX_SHORT_DURATION - accumulated_time
            if allowed_dur > 0.1:
                seg_end = seg_start + allowed_dur
                timeline.append({"start": seg_start, "end": seg_end})

                for w in seg["words"]:
                    if w.start < seg_end:
                        word_offset_start = w.start - seg_start
                        word_offset_end = min(w.end, seg_end) - seg_start
                        new_w_start = accumulated_time + word_offset_start
                        new_w_end = accumulated_time + word_offset_end
                        remapped_words.append({
                            "word": w.word,
                            "start": new_w_start,
                            "end": new_w_end
                        })
                accumulated_time += allowed_dur
            break
        else:
            timeline.append({"start": seg_start, "end": seg_end})

            for w in seg["words"]:
                word_offset_start = w.start - seg_start
                word_offset_end = w.end - seg_start
                new_w_start = accumulated_time + word_offset_start
                new_w_end = accumulated_time + word_offset_end
                remapped_words.append({
                    "word": w.word,
                    "start": new_w_start,
                    "end": new_w_end
                })

            accumulated_time += seg_dur

    print(f"Jump-cut silence removal applied: {original_duration:.2f}s -> {accumulated_time:.2f}s (capped at {MAX_SHORT_DURATION}s)")
    return timeline, remapped_words, accumulated_time


def fallback_metadata(broadcaster, raw_title):
    clean_title = re.sub(r"https?://\S+", "", raw_title)
    clean_title = re.sub(r"#\S+", "", clean_title)
    clean_title = "".join(c for c in clean_title if 32 <= ord(c) < 127).strip()
    clean_title = re.sub(r"\s+", " ", clean_title)
    # Remove stray quote characters
    clean_title = clean_title.replace('"', "").replace("'", "")

    clean_broadcaster = "".join(c for c in broadcaster if 32 <= ord(c) < 127).strip()

    # Strip duplicate streamer prefix
    if clean_broadcaster and clean_title.upper().startswith(clean_broadcaster.upper()):
        clean_title = clean_title[len(clean_broadcaster):].lstrip(":- ").strip()

    prefix = f"{clean_broadcaster.upper()}: " if clean_broadcaster else ""
    full_text = f"{prefix}{clean_title}"

    # Wrap at word boundaries, max 24 chars per line, max 2 lines
    wrapped_lines = textwrap.wrap(full_text.upper(), width=24)[:2]
    hook = "\\N".join(wrapped_lines) if wrapped_lines else full_text.upper()[:24]

    yt_title = f"{full_text[:75]} #Shorts"
    ig_caption = (
        f"{full_text}\n\n"
        f"GTA 6 action from {broadcaster}!\n\n"
        "#GTAVI #GTA6 #Gaming #TwitchClips #GamingClips #RockstarGames "
        "#Shorts #Reels #ViralGaming #Gamer #GrandTheftAuto #Gameplay"
    )

    return {
        "hook_title": hook,
        "yt_title": yt_title,
        "ig_caption": ig_caption
    }


def generate_ai_metadata(broadcaster, raw_title, transcript):
    """
    Calls Google AI Studio Gemini API (google-genai SDK) to generate viral
    hook_title, yt_title, and ig_caption. Tries gemini-3.8-flash first,
    then falls back to gemini-2.5-flash. If both fail, uses fallback_metadata().
    Uses v1beta API version to support JSON mode and newer endpoints cleanly.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key or not genai:
        print("Gemini API key not found or google-genai not imported. Using fallback metadata.")
        return fallback_metadata(broadcaster, raw_title)

    prompt = (
        f"You are an expert viral gaming content creator specializing in Grand Theft Auto VI (GTA 6) Shorts and Reels.\n\n"
        f"Twitch Streamer: {broadcaster}\n"
        f"Original Clip Title: {raw_title}\n"
        f"Spoken Transcript: {transcript}\n\n"
        f"Based on this context, return a strictly valid JSON object with EXACTLY these three fields:\n"
        f"1. 'hook_title': A bold, curiosity-driven 2-line on-screen hook for the middle seam of the video. "
        f"Maximum 48 characters total. Separate the two lines with a standard newline '\\n' or a pipe '|' (e.g. 'LINE 1\\nLINE 2' or 'LINE 1|LINE 2'). "
        f"Strictly uppercase standard ASCII letters, numbers, and basic punctuation only. NO emojis, NO underscores. "
        f"Write a complete, punchy 2-line hook (max 24 characters per line, 2 lines max) that finishes its thought "
        f"cleanly—never end mid-sentence or with an unclosed quote.\n"
        f"2. 'yt_title': A catchy, high-CTR YouTube Shorts title under 75 characters that sparks immense curiosity and ends with '#Shorts'.\n"
        f"3. 'ig_caption': A 2-line engaging Instagram Reels caption summarizing the funny or crazy moment with a call-to-action "
        f"(such as 'Follow for daily GTA 6 clips!'), followed by a blank line and 12-15 relevant high-reach gaming/GTA VI hashtags "
        f"(e.g. #GTAVI #GTA6 #GrandTheftAuto6 #TwitchClips #GamingClips #Reels).\n\n"
        f"Return ONLY the raw JSON object, without markdown formatting or code blocks."
    )

    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash"]

    for model_name in models_to_try:
        try:
            client = genai.Client(api_key=gemini_key, http_options={"api_version": "v1beta"})
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            text = response.text.strip()
            data = json.loads(text)

            raw_hook = data.get("hook_title", "").strip().upper()
            # Convert standard \n or pipe into ASS newline \N safely (never decoding raw \N in JSON)
            hook_title = raw_hook.replace("\\n", "\n").replace("|", "\n")
            hook_title = hook_title.replace("\r", "").replace("\n", "\\N")
            hook_title = "".join(c for c in hook_title if 32 <= ord(c) < 127)
            # Ensure max ~48 characters total
            if len(hook_title.replace("\\N", " ")) > 48:
                parts = hook_title.split("\\N")
                if len(parts) >= 2:
                    hook_title = f"{parts[0][:24]}\\N{parts[1][:24]}"
                else:
                    hook_title = hook_title[:48]

            yt_title = data.get("yt_title", "").strip()
            if not yt_title.endswith("#Shorts"):
                yt_title = f"{yt_title[:67]} #Shorts"
            else:
                if len(yt_title) > 75:
                    yt_title = f"{yt_title[:67]} #Shorts"

            ig_caption = data.get("ig_caption", "").strip()

            if not hook_title or not yt_title or not ig_caption:
                raise ValueError("Incomplete metadata fields from Gemini response.")

            print(f"Gemini AI ({model_name}) Generated Hook: {hook_title}")
            print(f"Gemini AI ({model_name}) Generated YT Title: {yt_title}")
            return {
                "hook_title": hook_title,
                "yt_title": yt_title,
                "ig_caption": ig_caption
            }

        except Exception as e:
            print(f"Gemini {model_name} failed: {type(e).__name__}: {e}")
            continue

    print("All Gemini models failed. Reverting to fallback metadata.")
    return fallback_metadata(broadcaster, raw_title)


def write_ass_subtitles(remapped_words, ass_path, total_duration, hook_text, has_webcam=True):
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Hook,Arial,62,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,5,2,5,50,50,0,1\n"
        "Style: Default,Arial,68,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,45,45,420,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = []

    # Hook banner across seam (y=680 if webcam split, or top area if fallback)
    seam_y = 680 if has_webcam else 300
    end_time_str = format_ass_time(total_duration)
    lines.append(f"Dialogue: 1,0:00:00.00,{end_time_str},Hook,,0,0,0,,{{\\pos(540,{seam_y})}}{hook_text}")

    # Spoken subtitles
    for w in remapped_words:
        w_text = w["word"].strip().upper() if isinstance(w, dict) else w.word.strip().upper()
        w_text = "".join(c for c in w_text if 32 <= ord(c) < 127)
        if not w_text:
            continue
        w_start = w["start"] if isinstance(w, dict) else w.start
        w_end = w["end"] if isinstance(w, dict) else w.end
        lines.append(f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},Default,,0,0,0,,{w_text}")

    Path(ass_path).write_text(header + "\n".join(lines), encoding="utf-8")


def render_split_vertical(raw_mp4, ass_path, out_mp4, timeline_segments, cam_box):
    """
    1. Concatenate jump-cut segments into joined.mp4.
    2. Render split-screen 1080x1920:
       - If cam_box:
         [0:v]crop=cw:ch:cx:cy,scale=1080:680[top];
         [0:v]scale=1080:1240:force_original_aspect_ratio=increase,crop=1080:1240[bottom];
         [top][bottom]vstack=inputs=2,ass=ass_path[outv]
       - If no cam_box (fallback):
         [0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,ass=ass_path[outv]
    """
    temp_dir = tempfile.mkdtemp(prefix="render_split_")
    try:
        segment_files = []
        for i, seg in enumerate(timeline_segments):
            seg_start = seg["start"]
            seg_dur = seg["end"] - seg["start"]
            if seg_dur <= 0.05:
                continue

            seg_file = os.path.join(temp_dir, f"seg_{i:03d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{seg_start:.3f}",
                "-i", str(raw_mp4),
                "-t", f"{seg_dur:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                seg_file
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            segment_files.append(seg_file)

        if not segment_files:
            raise RuntimeError("No valid segments rendered during jump-cut.")

        joined_mp4 = os.path.join(temp_dir, "joined.mp4")
        if len(segment_files) == 1:
            joined_mp4 = segment_files[0]
        else:
            concat_list = os.path.join(temp_dir, "concat.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for s in segment_files:
                    f.write(f"file '{Path(s).as_posix()}'\n")

            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                joined_mp4
            ]
            subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        escaped_ass = Path(ass_path).as_posix().replace(":", r"\:")

        if cam_box:
            cx, cy, cw, ch = cam_box
            filtergraph = (
                f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale=1080:680[top];"
                f"[0:v]scale=1080:1240:force_original_aspect_ratio=increase,crop=1080:1240[bottom];"
                f"[top][bottom]vstack=inputs=2,ass={escaped_ass}[outv]"
            )
        else:
            filtergraph = (
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,ass={escaped_ass}[outv]"
            )

        final_cmd = [
            "ffmpeg", "-y",
            "-i", joined_mp4,
            "-filter_complex", filtergraph,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_mp4)
        ]
        subprocess.run(final_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def upload_youtube(video_path, title):
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
    )
    youtube = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": f"{title[:85]} #Shorts" if not title.endswith("#Shorts") else title[:95],
            "description": f"{title}\n\n#Shorts #Gaming #GTAVI",
            "categoryId": "20",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute()
    print("Uploaded YouTube Video ID:", resp["id"])


def upload_instagram_reel(video_path, ig_caption, clip_id):
    """
    Uploads video to Instagram Reels using a temporary GitHub Release as the public video host.
    1. Creates temporary release on Anvillll/yt-shorts-pipeline (tag: temp-reel-{clip_id}-{ts})
    2. Uploads video asset and retrieves browser_download_url
    3. Initiates Reels media container on Instagram Graph API
    4. Polls status until FINISHED (up to 90s)
    5. Publishes Reel
    6. Finally block cleans up GitHub release and git tag.
    """
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    ig_user_id = os.environ.get("IG_USER_ID", "").strip()
    ig_access_token = os.environ.get("IG_ACCESS_TOKEN", "").strip()

    if not ig_user_id or not ig_access_token:
        print("Instagram credentials (IG_USER_ID / IG_ACCESS_TOKEN) not set. Skipping Instagram Reel upload.")
        return False

    if not github_token:
        print("GITHUB_TOKEN not provided. Cannot host temporary release for Instagram Reel. Skipping.")
        return False

    repo = os.environ.get("GITHUB_REPOSITORY", "Anvillll/yt-shorts-pipeline")
    headers_gh = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    tag_name = f"temp-reel-{clip_id}-{int(time.time())}"
    release_id = None
    browser_download_url = None

    try:
        # 1. Create temporary GitHub release
        print(f"Creating temporary GitHub release: {tag_name}")
        create_rel_res = requests.post(
            f"https://api.github.com/repos/{repo}/releases",
            headers=headers_gh,
            json={
                "tag_name": tag_name,
                "name": f"Temp Reel {clip_id}",
                "body": "Temporary release for Instagram Reels hosting.",
                "draft": False,
                "prerelease": True
            },
            timeout=20
        )
        if create_rel_res.status_code >= 400:
            print(f"GitHub release creation error HTTP {create_rel_res.status_code}: {create_rel_res.text}")
        create_rel_res.raise_for_status()
        rel_data = create_rel_res.json()
        release_id = rel_data["id"]
        upload_url = rel_data["upload_url"].split("{")[0]

        # 2. Upload video asset to release
        asset_name = f"{clip_id}_reel.mp4"
        upload_asset_url = f"{upload_url}?name={asset_name}"
        headers_upload = {
            "Authorization": f"Bearer {github_token}",
            "Content-Type": "video/mp4",
        }
        print(f"Uploading video asset to temporary release ({asset_name})...")
        with open(video_path, "rb") as f:
            up_res = requests.post(upload_asset_url, headers=headers_upload, data=f, timeout=120)
        if up_res.status_code >= 400:
            print(f"GitHub asset upload error HTTP {up_res.status_code}: {up_res.text}")
        up_res.raise_for_status()
        browser_download_url = up_res.json().get("browser_download_url")
        print(f"Temporary asset download URL: {browser_download_url}")

        # Wait 2 seconds for GitHub release asset CDN to propagate
        time.sleep(2)

        # Resolve 302 redirect to the direct CDN URL for Meta's crawler
        direct_video_url = browser_download_url
        try:
            head_res = requests.head(browser_download_url, allow_redirects=True, timeout=15)
            if head_res.status_code >= 400:
                print(f"GitHub asset HEAD error HTTP {head_res.status_code}: {head_res.text}")
            direct_video_url = head_res.url or browser_download_url
        except Exception as e:
            print(f"Could not resolve direct CDN URL ({e}). Falling back to release asset URL.")
        print(f"Resolved direct CDN video URL: {direct_video_url}")

        # 3. Create Reels container via Instagram Graph API
        print("Creating Instagram Reels media container...")
        container_url = f"https://graph.facebook.com/v21.0/{ig_user_id}/media"
        c_res = requests.post(
            container_url,
            data={
                "media_type": "REELS",
                "video_url": direct_video_url,
                "caption": ig_caption,
                "access_token": ig_access_token,
            },
            timeout=30
        )
        if c_res.status_code >= 400:
            print(f"Instagram container creation error HTTP {c_res.status_code}: {c_res.text}")
        c_res.raise_for_status()
        creation_id = c_res.json().get("id")
        print(f"Reels container created. Creation ID: {creation_id}")

        # 4. Poll status until FINISHED (up to 90 seconds)
        status_url = f"https://graph.facebook.com/v21.0/{creation_id}"
        poll_start = time.time()
        is_ready = False

        while time.time() - poll_start < 90:
            time.sleep(5)
            s_res = requests.get(
                status_url,
                params={"fields": "status_code", "access_token": ig_access_token},
                timeout=15
            )
            if s_res.status_code >= 400:
                print(f"Instagram status check error HTTP {s_res.status_code}: {s_res.text}")
            s_res.raise_for_status()
            status_code = s_res.json().get("status_code")
            print(f"Instagram container status: {status_code}")

            if status_code == "FINISHED":
                is_ready = True
                break
            elif status_code == "ERROR":
                raise RuntimeError(f"Instagram media processing failed: {s_res.text}")

        if not is_ready:
            raise TimeoutError("Timed out waiting for Instagram Reel container to finish processing.")

        # 5. Publish the Reel
        print("Publishing Reel to Instagram...")
        publish_url = f"https://graph.facebook.com/v21.0/{ig_user_id}/media_publish"
        p_res = requests.post(
            publish_url,
            data={"creation_id": creation_id, "access_token": ig_access_token},
            timeout=30
        )
        if p_res.status_code >= 400:
            print(f"Instagram publish error HTTP {p_res.status_code}: {p_res.text}")
        p_res.raise_for_status()
        published_id = p_res.json().get("id")
        print(f"Successfully published Instagram Reel! Reel ID: {published_id}")
        return True

    except Exception as e:
        print(f"Instagram Reel upload failed: {e}")
        return False

    finally:
        # 6. Cleanup temporary release and git tag
        if release_id:
            print(f"Deleting temporary release ID {release_id}...")
            try:
                requests.delete(f"https://api.github.com/repos/{repo}/releases/{release_id}", headers=headers_gh, timeout=15)
            except Exception as e:
                print(f"Failed to delete release {release_id}: {e}")

        print(f"Deleting temporary git tag {tag_name}...")
        try:
            requests.delete(f"https://api.github.com/repos/{repo}/git/refs/tags/{tag_name}", headers=headers_gh, timeout=15)
        except Exception as e:
            print(f"Failed to delete tag {tag_name}: {e}")


def process_and_upload_candidate(clip, can_upload_yt, can_upload_ig):
    clip_id = clip["id"]
    title = clip.get("title", "")
    broadcaster = clip.get("broadcaster_name", "")
    print(f"\nEvaluating candidate clip ({clip.get('view_count', 0)} views): {clip_id} - '{title}' by {broadcaster}")

    raw_mp4 = Path(f"{clip_id}_raw.mp4")
    ass_path = Path(f"{clip_id}.ass")
    out_mp4 = Path(f"{clip_id}_short.mp4")

    attempted_upload = False

    try:
        subprocess.run(["yt-dlp", "-f", "b[ext=mp4]/mp4", "-o", str(raw_mp4), clip["url"]], check=True)

        # 1. Strict Nudity & Adult-Content Guard
        if contains_nsfw(raw_mp4):
            print(f"WARNING: Clip {clip_id} flagged for NSFW/nudity content! Skipping.")
            mark_processed(clip_id)
            return False, False

        # 2. Audio transcription & English check
        words, info, transcript = transcribe_and_verify(raw_mp4)
        if words is None:
            print(f"Clip {clip_id} audio was not verified English. Skipping.")
            mark_processed(clip_id)
            return False, False

        # 3. Detect streamer webcam position (1080:680 box)
        cam_box = detect_streamer_cam(raw_mp4)

        # 4. Jump-cut silence removal & timeline remapping
        orig_dur = get_video_duration(raw_mp4)
        timeline_segments, remapped_words, total_dur = build_jumpcut_timeline(words, orig_dur)

        # 5. Gemini AI Metadata Generation (hook_title, yt_title, ig_caption)
        metadata = generate_ai_metadata(broadcaster, title, transcript)
        hook_title = metadata["hook_title"]
        yt_title = metadata["yt_title"]
        ig_caption = metadata["ig_caption"]

        # 6. Build hook banner text and write ASS subtitles (Hook style at y=680, Default style at MarginV=420)
        write_ass_subtitles(remapped_words, ass_path, total_dur, hook_title, has_webcam=(cam_box is not None))

        # 7. Render stacked split-screen vertical video (top 1080x680 webcam, bottom 1080x1240 gameplay)
        render_split_vertical(raw_mp4, ass_path, out_mp4, timeline_segments, cam_box)

        # Clip has passed checks, rendered successfully, and is ready for upload attempt
        attempted_upload = True
        any_upload_succeeded = False

        # 8. Upload to YouTube (if within daily limit)
        if can_upload_yt:
            try:
                print("Uploading to YouTube Shorts...")
                upload_youtube(out_mp4, yt_title)
                increment_daily_count("yt")
                any_upload_succeeded = True
            except Exception as e:
                print(f"YouTube upload failed: {e}")
        else:
            print("YouTube daily limit reached for today. Skipping YouTube upload.")

        # 9. Upload to Instagram Reels (if within daily limit)
        if can_upload_ig:
            try:
                print("Uploading to Instagram Reels...")
                if upload_instagram_reel(out_mp4, ig_caption, clip_id):
                    increment_daily_count("ig")
                    any_upload_succeeded = True
            except Exception as e:
                print(f"Instagram upload failed: {e}")
        else:
            print("Instagram daily limit reached for today. Skipping Instagram upload.")

        mark_processed(clip_id)
        return any_upload_succeeded, attempted_upload

    except Exception as e:
        print(f"Error processing clip {clip_id}: {e}")
        mark_processed(clip_id)
        return False, attempted_upload

    finally:
        for p in (raw_mp4, ass_path, out_mp4):
            if p.exists():
                p.unlink()


def main():
    yt_count, ig_count = get_daily_counts()
    print(f"Today's current upload count: YouTube={yt_count}/{MAX_YT_PER_DAY}, Instagram={ig_count}/{MAX_IG_PER_DAY}")

    can_upload_yt = yt_count < MAX_YT_PER_DAY
    can_upload_ig = ig_count < MAX_IG_PER_DAY

    if not can_upload_yt and not can_upload_ig:
        print(f"All daily upload caps reached for today (YouTube: {yt_count}/{MAX_YT_PER_DAY}, Instagram: {ig_count}/{MAX_IG_PER_DAY}). Exiting cleanly.")
        return

    token = get_twitch_token()
    game_id = get_gta6_game_id(token)
    if not game_id:
        print("Grand Theft Auto VI category not found on Twitch.")
        return

    uploaded = False
    upload_attempts = 0

    # Stage 1: Prioritize top streamers first
    print("\n=== Stage 1: Checking Priority Streamers for GTA VI Clips ===")
    try:
        broadcaster_ids = get_broadcaster_ids(token, PRIORITY_STREAMERS)
        print(f"Found {len(broadcaster_ids)} broadcaster IDs for priority streamers.")
    except Exception as e:
        print(f"Failed to fetch priority broadcaster IDs: {e}")
        broadcaster_ids = []

    if broadcaster_ids:
        for p_hours in [24, 48]:
            print(f"\n--- Checking Priority Streamers in the last {p_hours} hours ---")
            priority_clips = []
            for b_id in broadcaster_ids:
                try:
                    b_clips = fetch_clips_for_broadcaster(token, b_id, p_hours)
                    for c in b_clips:
                        if (
                            c.get("game_id") == game_id
                            and c.get("id") not in PROCESSED
                            and (10 <= c.get("duration", 0) <= 60)
                            and c.get("language") == "en"
                        ):
                            priority_clips.append(c)
                except Exception as e:
                    print(f"Error fetching clips for broadcaster {b_id}: {e}")

            # Sort priority clips by view_count descending
            priority_clips.sort(key=lambda c: c.get("view_count", 0), reverse=True)

            for clip in priority_clips:
                clip_id = clip["id"]
                if clip_id in PROCESSED:
                    continue

                title = clip.get("title", "")
                if not is_ascii_english_title(title):
                    print(f"Skipping priority clip {clip_id}: title failed ASCII English check ('{title}').")
                    mark_processed(clip_id)
                    continue

                if has_blocked_keywords(title):
                    print(f"Skipping priority clip {clip_id}: title contains blocked adult keyword ('{title}').")
                    mark_processed(clip_id)
                    continue

                success, attempted = process_and_upload_candidate(clip, can_upload_yt, can_upload_ig)
                if attempted:
                    upload_attempts += 1

                if success:
                    uploaded = True
                    print(f"Successfully processed and published priority streamer clip {clip_id} from {p_hours}h window!")
                    break

                if upload_attempts >= MAX_UPLOAD_ATTEMPTS_PER_RUN:
                    print("Reached max upload attempts for this run. Exiting cleanly.")
                    return

            if uploaded:
                break

    # Stage 2: General viral GTA VI clips fallback across progressive time windows
    if not uploaded:
        print("\n=== Stage 2: Falling back to General Viral GTA VI Clips ===")
        for hours in FALLBACK_HOURS:
            print(f"\n--- Searching Twitch clips in the last {hours} hours window ---")
            clips = fetch_clips_for_window(token, game_id, hours)

            for clip in clips:
                clip_id = clip["id"]
                if clip_id in PROCESSED:
                    continue

                title = clip.get("title", "")
                # Strict English check on metadata & title
                if clip.get("language") != "en":
                    print(f"Skipping clip {clip_id}: language is '{clip.get('language')}', not 'en'.")
                    mark_processed(clip_id)
                    continue

                if not is_ascii_english_title(title):
                    print(f"Skipping clip {clip_id}: title failed ASCII English check ('{title}').")
                    mark_processed(clip_id)
                    continue

                # Nudity & adult-content keyword blocklist
                if has_blocked_keywords(title):
                    print(f"Skipping clip {clip_id}: title contains blocked adult keyword ('{title}').")
                    mark_processed(clip_id)
                    continue

                duration = clip.get("duration", 0)
                if not (10 <= duration <= 60):
                    print(f"Skipping clip {clip_id}: duration {duration}s not between 10-60s.")
                    mark_processed(clip_id)
                    continue

                # Process candidate and upload to eligible platforms
                success, attempted = process_and_upload_candidate(clip, can_upload_yt, can_upload_ig)
                if attempted:
                    upload_attempts += 1

                if success:
                    uploaded = True
                    print(f"Successfully processed and published clip {clip_id} from {hours}h window!")
                    break

                if upload_attempts >= MAX_UPLOAD_ATTEMPTS_PER_RUN:
                    print("Reached max upload attempts for this run. Exiting cleanly.")
                    return

            if uploaded:
                break

            print(f"No clean, valid clips found in {hours}h window. Expanding search window...")

    if not uploaded:
        print("No valid GTA VI clips found across all progressive fallback windows.")


if __name__ == "__main__":
    main()
