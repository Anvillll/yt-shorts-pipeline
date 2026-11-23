import atexit
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
MAX_YT2_PER_DAY = 2  # 2 YouTube Shorts for Channel #2 per UTC day (Morning and Night IST runs)
MAX_YT3_PER_DAY = 2  # Channel #3 Minecraft Shorts per day (09:00 and 21:00 IST slots)
MAX_IG_PER_DAY = 3  # Up to 3 Instagram Reels per UTC day (Morning, Noon, Night)
MAX_UPLOAD_ATTEMPTS_PER_RUN = 3  # Max candidate clips to render/attempt per single CI run
FALLBACK_HOURS = [24, 48, 72, 168, 720]  # Progressive fallback windows
UPLOAD_TARGET = os.getenv("UPLOAD_TARGET", "all").strip().lower()
IST = timezone(timedelta(hours=5, minutes=30))
MC_SCHEDULE_FILE = Path("mc_schedule.json")
COOKIES_FILE = Path("cookies.txt")
_COOKIES_CREATED_FROM_ENV = False

MINECRAFT_CREATORS = ["@whereswrld", "@wemmbumc", "@DrDonutt", "@aCookieGod"]

PRIORITY_STREAMERS = [
    "kaicenat", "ishowspeed", "xqc", "caseoh_", "jynxzi",
    "adinross", "stableronaldo", "plaqueboymax"
]

BLOCKED_KEYWORDS = [
    "strip", "stripclub", "club", "nsfw", "nude", "nudity", "naked",
    "18+", "xxx", "porn", "hentai", "boobs", "booty", "lapdance",
    "unicorn", "vanilla", "bikini", "beach", "swimsuit", "lingerie",
    "underwear", "changing", "clothes", "outfit", "gyatt", "thicc",
    "baddie", "stripper",
]

EXPOSED_SENSITIVE_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

COVERED_SENSITIVE_CLASSES = {
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
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
            "-vf", "fps=3",
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
                if label in EXPOSED_SENSITIVE_CLASSES and score >= 0.28:
                    print(f"NSFW detected in frame {frame.name}: {label} ({score:.2f})")
                    return True
                if label in COVERED_SENSITIVE_CLASSES and score >= 0.55:
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
    aspect ratio. Otherwise returns None so callers can use a clean full-frame
    vertical layout (never a blind corner/HUD crop).
    Returns None if the video cannot be opened at all.
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
            print(f"Warning: Could not load any Haar cascade. Falling back to full-screen vertical framing.")
            cap.release()
            return None

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
            print(f"Streamer cam detection: Only {len(side_faces)}/16 side faces detected. Using full-frame vertical layout.")
            return None

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


def detect_facecam_safe(video_path):
    """
    Safe facecam wrapper. Returns a validated (x, y, w, h) box or None.
    Never raises, and never returns a blind corner/HUD crop when no confident
    facecam is present, so faceless videos fall back to a clean full-frame layout.
    """
    try:
        box = detect_streamer_cam(video_path)
    except Exception as e:
        print(f"Facecam detection error ({e}); using full-frame vertical layout.")
        return None

    if not box or len(box) != 4:
        print("No confident facecam detected; using full-frame vertical layout.")
        return None

    try:
        x, y, w, h = (int(v) for v in box)
    except (TypeError, ValueError):
        print("Invalid facecam box; using full-frame vertical layout.")
        return None

    if w < 160 or h < 100:
        print(f"Facecam box too small ({w}x{h}); using full-frame vertical layout.")
        return None

    frame_w = frame_h = 0
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception as e:
        print(f"Could not validate facecam bounds ({e}); using full-frame vertical layout.")
        return None

    if frame_w <= 0 or frame_h <= 0:
        print("Could not read video dimensions; using full-frame vertical layout.")
        return None

    x = max(0, min(x, frame_w - 2))
    y = max(0, min(y, frame_h - 2))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)

    if w < 160 or h < 100:
        print("Facecam box exceeds frame bounds; using full-frame vertical layout.")
        return None

    return (x, y, w, h)


def get_daily_counts():
    """
    Reads daily_count.txt supporting:
    - 'YYYY-MM-DD:yt=X,yt2=Z,ig=Y'
    - 'YYYY-MM-DD:yt=X,ig=Y' (yt2 defaults to 0)
    - 'YYYY-MM-DD:X' (legacy format treated as yt=X, yt2=0, ig=0)
    Returns (yt_count, yt2_count, ig_count) for today's UTC date.
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count_file = Path("daily_count.txt")
    if not count_file.exists():
        return 0, 0, 0

    content = count_file.read_text().strip()
    if ":" not in content:
        return 0, 0, 0

    saved_date, counts_part = content.split(":", 1)
    if saved_date != today_str:
        return 0, 0, 0

    yt_count = 0
    yt2_count = 0
    ig_count = 0

    if "yt=" in counts_part or "yt2=" in counts_part or "ig=" in counts_part:
        for item in counts_part.split(","):
            item = item.strip()
            if item.startswith("yt="):
                try:
                    yt_count = int(item.split("=")[1])
                except ValueError:
                    pass
            elif item.startswith("yt2="):
                try:
                    yt2_count = int(item.split("=")[1])
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

    return yt_count, yt2_count, ig_count


def increment_daily_count(platform):
    """
    Increments 'yt', 'yt2', or 'ig' counter for today's UTC date.
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yt_count, yt2_count, ig_count = get_daily_counts()

    if platform == "yt":
        yt_count += 1
    elif platform == "yt2":
        yt2_count += 1
    elif platform == "ig":
        ig_count += 1

    count_file = Path("daily_count.txt")
    count_file.write_text(f"{today_str}:yt={yt_count},yt2={yt2_count},ig={ig_count}", encoding="utf-8")
    print(f"Updated daily count for {today_str}: yt={yt_count}, yt2={yt2_count}, ig={ig_count}")


def is_yt2_time_window():
    """
    Channel #2 uploads only run on the Morning (< 12:00 IST) and Night (>= 18:00 IST)
    runs. The Noon (12:00-17:59 IST) run is skipped.
    """
    now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return now_ist.hour < 12 or now_ist.hour >= 18


def ist_now():
    return datetime.now(IST)


def utc_now():
    return datetime.now(timezone.utc)


def parse_utc(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def load_mc_schedule():
    schedule = {"yt3_last_scheduled_utc": None, "yt2_morning_scheduled_dates": []}
    try:
        if MC_SCHEDULE_FILE.exists():
            data = json.loads(MC_SCHEDULE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                schedule["yt3_last_scheduled_utc"] = data.get("yt3_last_scheduled_utc")
                dates = data.get("yt2_morning_scheduled_dates") or []
                if isinstance(dates, list):
                    schedule["yt2_morning_scheduled_dates"] = [str(d) for d in dates]
    except Exception as e:
        print(f"Warning: Could not read {MC_SCHEDULE_FILE} ({e}). Starting fresh.")
    return schedule


def save_mc_schedule(schedule):
    try:
        MC_SCHEDULE_FILE.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
        print(f"Saved Minecraft schedule: {schedule}")
    except Exception as e:
        print(f"Warning: Could not write {MC_SCHEDULE_FILE}: {e}")


def yt3_queue_is_resting(schedule):
    last = parse_utc(schedule.get("yt3_last_scheduled_utc"))
    if last is None:
        return False
    return last > utc_now() + timedelta(minutes=30)


def next_ist_slots(count, morning_only=False):
    """Return the next `count` 09:00/21:00 IST slots strictly in the future."""
    now = ist_now()
    slots = []
    day = now.date()
    while len(slots) < count:
        times = [(9, 0)] if morning_only else [(9, 0), (21, 0)]
        for hh, mm in times:
            slot = datetime(day.year, day.month, day.day, hh, mm, tzinfo=IST)
            if slot > now + timedelta(minutes=2):
                slots.append(slot)
                if len(slots) >= count:
                    break
        day = day + timedelta(days=1)
    return slots


def next_morning_ist_slot(day_offset=0):
    now = ist_now()
    base = now.date() + timedelta(days=day_offset)
    return datetime(base.year, base.month, base.day, 9, 0, tzinfo=IST)


def to_utc_iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _remove_runtime_cookies():
    if not _COOKIES_CREATED_FROM_ENV:
        return
    try:
        if COOKIES_FILE.exists():
            COOKIES_FILE.unlink()
            print("Removed runtime cookies.txt.")
    except Exception as e:
        print(f"Warning: Could not remove cookies.txt: {e}")


def _ensure_cookies_file():
    global _COOKIES_CREATED_FROM_ENV
    if COOKIES_FILE.exists():
        return
    data = os.getenv("YT_COOKIES", "")
    if not data.strip():
        return
    try:
        COOKIES_FILE.write_text(
            data if data.endswith("\n") else data + "\n",
            encoding="utf-8",
        )
        _COOKIES_CREATED_FROM_ENV = True
        atexit.register(_remove_runtime_cookies)
        print("Created cookies.txt from YT_COOKIES secret.")
    except Exception as e:
        print(f"Warning: Could not create cookies.txt: {e}")


_YTDLP_SUPPORTS_REMOTE_COMPONENTS = None


def _ytdlp_supports_remote_components():
    global _YTDLP_SUPPORTS_REMOTE_COMPONENTS
    if _YTDLP_SUPPORTS_REMOTE_COMPONENTS is None:
        try:
            res = subprocess.run(
                ["yt-dlp", "--help"],
                capture_output=True, text=True, timeout=30
            )
            _YTDLP_SUPPORTS_REMOTE_COMPONENTS = "--remote-components" in (
                res.stdout + res.stderr
            )
        except Exception:
            _YTDLP_SUPPORTS_REMOTE_COMPONENTS = False
    return _YTDLP_SUPPORTS_REMOTE_COMPONENTS


def _youtube_auth_args(url):
    if not url or ("youtube.com" not in url and "youtu.be" not in url):
        return []
    args = ["--js-runtimes", "node"]
    if _ytdlp_supports_remote_components():
        args += ["--remote-components", "ejs:github"]
    try:
        if COOKIES_FILE.exists() and COOKIES_FILE.stat().st_size > 0:
            args = ["--cookies", str(COOKIES_FILE), "--sleep-requests", "1", *args]
    except Exception as e:
        print(f"Warning: Could not check cookies.txt: {e}")
    return args


_ensure_cookies_file()


def _yt_dlp_json(args, timeout=180):
    url = next(
        (
            arg for arg in args
            if isinstance(arg, str) and ("youtube.com" in arg or "youtu.be" in arg)
        ),
        "",
    )
    auth_args = _youtube_auth_args(url)
    try:
        res = subprocess.run(
            ["yt-dlp", "--no-warnings", *auth_args, *args],
            capture_output=True, text=True, timeout=timeout
        )
        if res.returncode != 0:
            print(f"yt-dlp failed: {res.stderr.strip()[:300]}")
            return None
        return json.loads(res.stdout)
    except Exception as e:
        print(f"yt-dlp error: {e}")
        return None


def _parse_upload_ts(entry):
    ts = entry.get("timestamp") or entry.get("release_timestamp")
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    upload_date = entry.get("upload_date")
    if upload_date:
        try:
            return datetime.strptime(str(upload_date), "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def select_minecraft_video(processed_ids):
    """
    Select a long-form Minecraft video using strict creator priority order.
    Uses yt-dlp flat playlist extraction (latest 6 videos per creator).
    Filters out Shorts (< 180s) and videos already present in processed.txt.
    Pass 1: first creator (priority order) with an unused video from the last 7
    days -> highest view_count. Pass 2: first creator with any unused long-form
    video -> most recent.
    """
    recent_cutoff = utc_now() - timedelta(days=7)
    creator_candidates = []

    for handle in MINECRAFT_CREATORS:
        url = f"https://www.youtube.com/{handle}/videos"
        data = _yt_dlp_json(["--flat-playlist", "--dump-single-json", "--playlist-end", "6", url])
        entries = (data or {}).get("entries") or []
        candidates = []
        for entry in entries[:6]:
            if not isinstance(entry, dict):
                continue
            video_id = entry.get("id")
            duration = entry.get("duration")
            if not video_id or video_id in processed_ids:
                continue
            if duration is not None and duration < 180:
                continue
            entry_url = entry.get("url") or ""
            if not entry_url.startswith("http"):
                entry_url = f"https://www.youtube.com/watch?v={video_id}"
            candidates.append({
                "id": video_id,
                "url": entry_url,
                "title": entry.get("title") or "",
                "duration": duration,
                "view_count": entry.get("view_count") or 0,
                "timestamp": _parse_upload_ts(entry),
                "creator": handle,
            })
        creator_candidates.append((handle, candidates))

    for handle, candidates in creator_candidates:
        recent = [
            c for c in candidates
            if c["timestamp"] is not None and c["timestamp"] >= recent_cutoff
        ]
        if recent:
            pick = max(recent, key=lambda c: c["view_count"])
            print(f"Minecraft selection: {handle} -> {pick['id']} (recent, {pick['view_count']} views)")
            return pick

    for handle, candidates in creator_candidates:
        if candidates:
            dated = [c for c in candidates if c["timestamp"] is not None]
            pick = max(dated, key=lambda c: c["timestamp"]) if dated else candidates[0]
            print(f"Minecraft selection (fallback): {handle} -> {pick['id']}")
            return pick

    print("No unused Minecraft long-form videos found for any priority creator.")
    return None


def fetch_video_details(video_url):
    data = _yt_dlp_json(["--skip-download", "-J", "--no-playlist", video_url])
    return data or {}


def _gemini_pick_windows(title, description, duration, need):
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key or not genai or need <= 0:
        return []

    prompt = (
        "You are selecting highlight windows from a Minecraft YouTube video.\n"
        f"Video title: {title}\n"
        f"Video description: {description[:3000]}\n"
        f"Total duration: {duration:.1f} seconds.\n"
        f"Pick exactly {need} non-overlapping highlight windows, each 45-58 seconds long, "
        "that are likely the most engaging, funny, or intense moments.\n"
        'Return ONLY raw JSON: {"windows": [{"start_sec": 0.0, "end_sec": 50.0}]}\n'
    )

    for model_name in ["gemini-2.5-flash", "gemini-3.8-flash"]:
        try:
            client = genai.Client(api_key=gemini_key, http_options={"api_version": "v1beta"})
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text.strip())
            windows = []
            for item in data.get("windows") or []:
                start = float(item.get("start_sec", 0))
                end = float(item.get("end_sec", 0))
                if 0 <= start < end <= duration and 45 <= (end - start) <= 58:
                    windows.append((start, end))
            if windows:
                print(f"Gemini ({model_name}) selected {len(windows)} Minecraft windows.")
                return windows
        except Exception as e:
            print(f"Gemini window selection {model_name} failed: {type(e).__name__}: {e}")
    return []


def pick_minecraft_windows(video_info, count=4):
    duration = float(video_info.get("duration") or 0)
    if duration <= 0:
        return []

    win = 52.0
    max_fit = (duration / max(1, count)) - 2.0
    if max_fit < win:
        win = max(45.0, max_fit)

    if duration < 45:
        return [(0.0, max(10.0, duration))]

    windows = []
    heatmap = video_info.get("heatmap") or []

    scored = []
    if heatmap:
        points = []
        for seg in heatmap:
            try:
                start = float(seg.get("start_time", 0))
                end = float(seg.get("end_time", start))
                value = float(seg.get("value", 0))
            except (TypeError, ValueError):
                continue
            if end > start:
                points.append((start, end, value))
        if points:
            t = 0.0
            while t + win <= duration + 0.01:
                score = 0.0
                for start, end, value in points:
                    overlap = max(0.0, min(t + win, end) - max(t, start))
                    if overlap > 0:
                        score += value * overlap
                scored.append((score, t))
                t += 5.0
    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        for _, start in scored:
            end = min(duration, start + win)
            if end - start < 45:
                continue
            if all(end <= s or start >= e for s, e in windows):
                windows.append((start, end))
            if len(windows) >= count:
                break

    if len(windows) < count:
        need = count - len(windows)
        for start, end in _gemini_pick_windows(
            video_info.get("title") or "",
            video_info.get("description") or "",
            duration,
            need,
        ):
            if all(end <= s or start >= e for s, e in windows):
                windows.append((start, end))
            if len(windows) >= count:
                break

    if len(windows) < count:
        span = duration / count
        for i in range(count):
            start = max(0.0, i * span + max(0.0, (span - win) / 2.0))
            end = min(duration, start + win)
            if end - start < 45:
                continue
            if all(end <= s or start >= e for s, e in windows):
                windows.append((start, end))
            if len(windows) >= count:
                break

    print(f"Minecraft windows selected: {[(round(s, 1), round(e, 1)) for s, e in windows[:count]]}")
    return windows[:count]


def download_video_slice(video_url, start_sec, end_sec, out_path):
    fmt = "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"
    cmd = [
        "yt-dlp", "--no-warnings", "--no-playlist",
        *_youtube_auth_args(video_url),
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start_sec:.3f}-{end_sec:.3f}",
        "--force-keyframes-at-cuts",
        "-o", str(out_path),
        video_url,
    ]
    subprocess.run(cmd, check=True)


def _clean_hook(raw):
    raw_hook = (raw or "").strip().upper()
    hook_title = raw_hook.replace("\\n", "\n").replace("|", "\n")
    hook_title = hook_title.replace("\r", "").replace("\n", "\\N")
    hook_title = "".join(c for c in hook_title if 32 <= ord(c) < 127)
    if len(hook_title.replace("\\N", " ")) > 48:
        parts = hook_title.split("\\N")
        if len(parts) >= 2:
            hook_title = f"{parts[0][:24]}\\N{parts[1][:24]}"
        else:
            hook_title = hook_title[:48]
    return hook_title


def _clean_yt_title(raw):
    yt_title = (raw or "").strip()
    if not yt_title.endswith("#Shorts"):
        yt_title = f"{yt_title[:67]} #Shorts"
    elif len(yt_title) > 75:
        yt_title = f"{yt_title[:67]} #Shorts"
    return yt_title


def fallback_minecraft_metadata(creator, raw_title):
    clean_title = re.sub(r"https?://\S+", "", raw_title or "")
    clean_title = re.sub(r"#\S+", "", clean_title)
    clean_title = "".join(c for c in clean_title if 32 <= ord(c) < 127).strip()
    clean_title = re.sub(r"\s+", " ", clean_title).replace('"', "").replace("'", "")
    full_text = clean_title or "MINECRAFT MOMENT"

    wrapped_lines = textwrap.wrap(full_text.upper(), width=24)[:2]
    hook = "\\N".join(wrapped_lines) if wrapped_lines else full_text.upper()[:24]

    yt_title = f"{full_text[:75]} #Shorts"
    description = (
        f"{full_text}\n\n"
        f"Minecraft gameplay from {creator}!\n\n"
        "#Minecraft #MinecraftShorts #MinecraftClips #Gaming #GamingShorts "
        "#Shorts #ViralGaming #Gamer #SurvivalMinecraft #MinecraftBuilds"
    )
    return {"hook_title": hook, "yt_title": yt_title, "description": description}


def generate_minecraft_metadata(creator, raw_title, transcript):
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key or not genai:
        return fallback_minecraft_metadata(creator, raw_title)

    prompt = (
        "You are an expert viral Minecraft Shorts creator.\n\n"
        f"Creator: {creator}\n"
        f"Original Video Title: {raw_title}\n"
        f"Spoken Transcript: {transcript}\n\n"
        "Return a strictly valid JSON object with EXACTLY these three fields:\n"
        "1. 'hook_title': a bold 2-line on-screen hook, max 48 characters total, "
        "two lines separated by a standard newline '\\n' or a pipe '|', uppercase "
        "ASCII letters/numbers/basic punctuation only, no emojis or underscores.\n"
        "2. 'yt_title': a catchy high-CTR YouTube Shorts title under 75 characters "
        "ending with '#Shorts'.\n"
        "3. 'description': a short 2-line description with a call-to-action followed "
        "by 10-15 Minecraft/gaming hashtags.\n\n"
        "Return ONLY the raw JSON object, without markdown formatting or code blocks."
    )

    for model_name in ["gemini-2.5-flash", "gemini-3.8-flash"]:
        try:
            client = genai.Client(api_key=gemini_key, http_options={"api_version": "v1beta"})
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(response.text.strip())

            hook_title = _clean_hook(data.get("hook_title", ""))
            yt_title = _clean_yt_title(data.get("yt_title", ""))
            description = (data.get("description") or "").strip()
            if not description:
                description = (
                    f"{yt_title}\n\n"
                    "#Minecraft #MinecraftShorts #MinecraftClips #Gaming #Shorts"
                )

            if not hook_title or not yt_title:
                raise ValueError("Incomplete Minecraft metadata fields from Gemini response.")

            print(f"Gemini AI ({model_name}) Minecraft Hook: {hook_title}")
            print(f"Gemini AI ({model_name}) Minecraft Title: {yt_title}")
            return {
                "hook_title": hook_title,
                "yt_title": yt_title,
                "description": description,
            }
        except Exception as e:
            print(f"Gemini {model_name} Minecraft metadata failed: {type(e).__name__}: {e}")
            continue

    print("All Gemini models failed for Minecraft metadata. Using fallback.")
    return fallback_minecraft_metadata(creator, raw_title)


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
    hook_title, yt_title, and ig_caption. Tries gemini-2.5-flash first,
    then falls back to gemini-3.8-flash. If both fail, uses fallback_metadata().
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

    models_to_try = ["gemini-2.5-flash", "gemini-3.8-flash"]

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
       - If no cam_box (faceless fallback):
         [0:v]split=2[bg][fg];
         [bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:2[bgblur];
         [fg]scale=1080:-2[fgscaled];
         [bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2,ass=ass_path[outv]
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
                f"[0:v]split=2[bg][fg];"
                f"[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:2[bgblur];"
                f"[fg]scale=1080:-2[fgscaled];"
                f"[bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2,ass={escaped_ass}[outv]"
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


def upload_youtube(
    video_path,
    title,
    refresh_token=None,
    channel_label="Primary",
    privacy_status="public",
    publish_at=None,
    description=None,
):
    refresh_token = refresh_token or os.getenv("YT_REFRESH_TOKEN")
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
    )
    youtube = build("youtube", "v3", credentials=creds)

    status = {
        "privacyStatus": privacy_status,
        "selfDeclaredMadeForKids": False,
    }
    if publish_at:
        # YouTube requires scheduled uploads to be private with a future publishAt.
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at

    body = {
        "snippet": {
            "title": f"{title[:85]} #Shorts" if not title.endswith("#Shorts") else title[:95],
            "description": description or f"{title}\n\n#Shorts #Gaming #GTAVI",
            "categoryId": "20",
        },
        "status": status,
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = req.execute()
    print(f"Uploaded YouTube Video ID ({channel_label}):", resp["id"])
    return resp["id"]


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


def run_minecraft_batch(schedule, clip_count=4, upload_yt3=True, upload_yt2=False, yt2_count=0):
    """
    Select, render, and distribute a Minecraft batch without any NSFW scan.
    Channel #3 receives Clip 1 public + Clips 2-4 scheduled (private/publishAt).
    When upload_yt2 is set, reuse rendered clips for upcoming Channel #2 Morning
    IST slots (no re-rendering).
    """
    video = select_minecraft_video(PROCESSED)
    if not video:
        return False

    print(f"Selected Minecraft video: {video['id']} - '{video['title']}' ({video['creator']})")
    info = fetch_video_details(video["url"])
    if info:
        video["title"] = info.get("title") or video["title"]
        video["duration"] = info.get("duration") or video.get("duration")
        video["creator"] = info.get("uploader") or video.get("creator")
        video["view_count"] = info.get("view_count") or video.get("view_count")
    if not info:
        info = {}
    info.setdefault("duration", video.get("duration") or 0)

    duration = float(video.get("duration") or info.get("duration") or 0)
    if duration and duration < 180:
        print(f"Minecraft video {video['id']} is a Short (<180s). Marking processed and skipping.")
        mark_processed(video["id"])
        return False

    windows = pick_minecraft_windows(info, count=clip_count)
    if not windows:
        print("Could not select Minecraft clip windows. Skipping batch.")
        return False

    rendered = []
    temp_paths = []
    uploaded_any = False

    try:
        for i, (start_sec, end_sec) in enumerate(windows):
            slice_path = Path(f"mc_{video['id']}_{i + 1}_slice.mp4")
            ass_path = Path(f"mc_{video['id']}_{i + 1}.ass")
            out_path = Path(f"mc_{video['id']}_{i + 1}_short.mp4")
            temp_paths.extend([slice_path, ass_path])

            try:
                print(f"Downloading Minecraft slice {i + 1}/{len(windows)}: {start_sec:.1f}-{end_sec:.1f}s")
                download_video_slice(video["url"], start_sec, end_sec, slice_path)
                dur = get_video_duration(slice_path)

                words, _info, transcript = transcribe_and_verify(slice_path)
                if words is None:
                    words = []
                    transcript = video.get("title", "")

                remapped_words = [
                    {"word": w.word, "start": max(0.0, w.start), "end": min(w.end, dur)}
                    for w in words if w.start < dur
                ]
                cam_box = detect_facecam_safe(slice_path)
                metadata = generate_minecraft_metadata(
                    video.get("creator") or video["creator"],
                    video.get("title") or "",
                    transcript,
                )
                write_ass_subtitles(
                    remapped_words, ass_path, dur, metadata["hook_title"],
                    has_webcam=bool(cam_box),
                )
                render_split_vertical(
                    slice_path, ass_path, out_path,
                    [{"start": 0.0, "end": dur}], cam_box,
                )
                rendered.append({
                    "path": out_path,
                    "title": metadata["yt_title"],
                    "description": metadata["description"],
                    "start": start_sec,
                    "end": end_sec,
                })
                print(f"Rendered Minecraft clip {i + 1}: {out_path}")
            except Exception as e:
                print(f"Minecraft slice {i + 1} failed: {e}")
                continue

        if not rendered:
            print("No Minecraft clips rendered successfully.")
            return False

        yt3_token = os.getenv("YT_REFRESH_TOKEN_3", "").strip()

        if upload_yt3 and yt3_token:
            # Clip 1: publish immediately.
            try:
                video_id = upload_youtube(
                    rendered[0]["path"],
                    rendered[0]["title"],
                    refresh_token=yt3_token,
                    channel_label="Channel #3",
                    privacy_status="public",
                    description=rendered[0]["description"],
                )
                uploaded_any = True
                print(f"Channel #3 immediate upload: {video_id}")
            except Exception as e:
                print(f"Channel #3 immediate upload failed: {e}")

            # Clips 2-4: schedule at the next 09:00/21:00 IST slots.
            scheduled_clips = rendered[1:4]
            if scheduled_clips:
                slots = next_ist_slots(len(scheduled_clips))
                last_scheduled = None
                for clip, slot in zip(scheduled_clips, slots):
                    publish_at = to_utc_iso(slot)
                    try:
                        video_id = upload_youtube(
                            clip["path"],
                            clip["title"],
                            refresh_token=yt3_token,
                            channel_label="Channel #3",
                            privacy_status="private",
                            publish_at=publish_at,
                            description=clip["description"],
                        )
                        uploaded_any = True
                        last_scheduled = publish_at
                        print(f"Channel #3 scheduled upload: {video_id} at {publish_at}")
                    except Exception as e:
                        print(f"Channel #3 scheduled upload failed for {publish_at}: {e}")
                if last_scheduled:
                    schedule["yt3_last_scheduled_utc"] = last_scheduled
                    save_mc_schedule(schedule)
                    print(f"Channel #3 queue set through {last_scheduled}")

            # Reuse rendered clips for Channel #2 Morning IST slots.
            if upload_yt2 and os.getenv("YT_REFRESH_TOKEN_2"):
                now_ist = ist_now()
                today_ist = now_ist.strftime("%Y-%m-%d")
                dates = [
                    d for d in schedule.get("yt2_morning_scheduled_dates", [])
                    if d >= today_ist
                ]

                if now_ist.hour < 12:
                    if yt2_count == 0:
                        try:
                            upload_youtube(
                                rendered[0]["path"],
                                rendered[0]["title"],
                                refresh_token=os.getenv("YT_REFRESH_TOKEN_2"),
                                channel_label="Channel #2",
                                privacy_status="public",
                                description=rendered[0]["description"],
                            )
                            increment_daily_count("yt2")
                            uploaded_any = True
                            if today_ist not in dates:
                                dates.append(today_ist)
                            print("Channel #2 Morning Minecraft upload complete (Clip 1).")
                        except Exception as e:
                            print(f"Channel #2 Morning upload failed: {e}")

                    if len(rendered) >= 3:
                        slot = next_morning_ist_slot(1)
                        slot_date = slot.strftime("%Y-%m-%d")
                        try:
                            upload_youtube(
                                rendered[2]["path"],
                                rendered[2]["title"],
                                refresh_token=os.getenv("YT_REFRESH_TOKEN_2"),
                                channel_label="Channel #2",
                                privacy_status="private",
                                publish_at=to_utc_iso(slot),
                                description=rendered[2]["description"],
                            )
                            uploaded_any = True
                            if slot_date not in dates:
                                dates.append(slot_date)
                            print(f"Channel #2 scheduled Clip 3 for {slot_date} 09:00 IST.")
                        except Exception as e:
                            print(f"Channel #2 scheduled upload failed: {e}")

                elif now_ist.hour >= 18 and len(rendered) >= 4:
                    for clip_index, day_offset in ((1, 1), (3, 2)):
                        slot = next_morning_ist_slot(day_offset)
                        slot_date = slot.strftime("%Y-%m-%d")
                        try:
                            upload_youtube(
                                rendered[clip_index]["path"],
                                rendered[clip_index]["title"],
                                refresh_token=os.getenv("YT_REFRESH_TOKEN_2"),
                                channel_label="Channel #2",
                                privacy_status="private",
                                publish_at=to_utc_iso(slot),
                                description=rendered[clip_index]["description"],
                            )
                            uploaded_any = True
                            if slot_date not in dates:
                                dates.append(slot_date)
                            print(f"Channel #2 scheduled Clip {clip_index + 1} for {slot_date} 09:00 IST.")
                        except Exception as e:
                            print(f"Channel #2 scheduled upload failed for {slot_date}: {e}")

                schedule["yt2_morning_scheduled_dates"] = sorted(set(dates))
                save_mc_schedule(schedule)

        elif upload_yt2 and os.getenv("YT_REFRESH_TOKEN_2"):
            # Single immediate Channel #2 Minecraft upload (manual test / no Channel #3).
            try:
                upload_youtube(
                    rendered[0]["path"],
                    rendered[0]["title"],
                    refresh_token=os.getenv("YT_REFRESH_TOKEN_2"),
                    channel_label="Channel #2",
                    privacy_status="public",
                    description=rendered[0]["description"],
                )
                increment_daily_count("yt2")
                uploaded_any = True
                print("Channel #2 Minecraft upload complete (Clip 1).")
            except Exception as e:
                print(f"Channel #2 Minecraft upload failed: {e}")

        if uploaded_any:
            mark_processed(video["id"])

        return uploaded_any

    finally:
        for path in temp_paths + [item["path"] for item in rendered]:
            try:
                if Path(path).exists():
                    Path(path).unlink()
            except Exception:
                pass


def process_and_upload_candidate(clip, can_upload_yt, can_upload_yt2, can_upload_ig):
    clip_id = clip["id"]
    title = clip.get("title", "")
    broadcaster = clip.get("broadcaster_name", "")
    print(f"\nEvaluating candidate clip ({clip.get('view_count', 0)} views): {clip_id} - '{title}' by {broadcaster}")

    raw_mp4 = Path(f"{clip_id}_raw.mp4")
    ass_path = Path(f"{clip_id}.ass")
    out_mp4 = Path(f"{clip_id}_short.mp4")

    attempted_upload = False

    try:
        subprocess.run(["yt-dlp", *_youtube_auth_args(clip["url"]), "-f", "b[ext=mp4]/mp4", "-o", str(raw_mp4), clip["url"]], check=True)

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

        # 3. Safe streamer webcam detection (None -> clean full-frame fallback)
        cam_box = detect_facecam_safe(raw_mp4)

        # 4. Jump-cut silence removal & timeline remapping
        orig_dur = get_video_duration(raw_mp4)
        timeline_segments, remapped_words, total_dur = build_jumpcut_timeline(words, orig_dur)

        # 5. Gemini AI Metadata Generation (hook_title, yt_title, ig_caption)
        metadata = generate_ai_metadata(broadcaster, title, transcript)
        hook_title = metadata["hook_title"]
        yt_title = metadata["yt_title"]
        ig_caption = metadata["ig_caption"]

        # 6. Build hook banner text and write ASS subtitles (Hook style at y=680, Default style at MarginV=420)
        write_ass_subtitles(remapped_words, ass_path, total_dur, hook_title, has_webcam=bool(cam_box))

        # 7. Render stacked split-screen vertical video (top 1080x680 webcam, bottom 1080x1240 gameplay)
        render_split_vertical(raw_mp4, ass_path, out_mp4, timeline_segments, cam_box)

        # Clip has passed checks, rendered successfully, and is ready for upload attempt
        attempted_upload = True
        any_upload_succeeded = False

        # 8. Upload to YouTube Channel #1 (if within daily limit)
        if can_upload_yt:
            try:
                print("Uploading to YouTube Shorts (Primary channel)...")
                upload_youtube(out_mp4, yt_title, channel_label="Primary")
                increment_daily_count("yt")
                any_upload_succeeded = True
            except Exception as e:
                print(f"YouTube upload failed: {e}")
        else:
            print("YouTube daily limit reached for today. Skipping YouTube upload.")

        # 9. Upload the exact same rendered Short to YouTube Channel #2 (Morning/Night IST runs only)
        if can_upload_yt2:
            try:
                print("Uploading to YouTube Shorts (Channel #2)...")
                upload_youtube(
                    out_mp4,
                    yt_title,
                    refresh_token=os.getenv("YT_REFRESH_TOKEN_2"),
                    channel_label="Channel #2",
                )
                increment_daily_count("yt2")
                any_upload_succeeded = True
            except Exception as e:
                print(f"YouTube Channel #2 upload failed: {e}")
        else:
            print("YouTube Channel #2 not eligible (token missing, daily cap reached, or outside IST window). Skipping.")

        # 10. Upload to Instagram Reels (if within daily limit)
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


def run_gta_pipeline(can_upload_yt, can_upload_yt2, can_upload_ig):
    token = get_twitch_token()
    game_id = get_gta6_game_id(token)
    if not game_id:
        print("Grand Theft Auto VI category not found on Twitch.")
        return False

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

                success, attempted = process_and_upload_candidate(clip, can_upload_yt, can_upload_yt2, can_upload_ig)
                if attempted:
                    upload_attempts += 1

                if success:
                    uploaded = True
                    print(f"Successfully processed and published priority streamer clip {clip_id} from {p_hours}h window!")
                    break

                if upload_attempts >= MAX_UPLOAD_ATTEMPTS_PER_RUN:
                    print("Reached max upload attempts for this run. Exiting cleanly.")
                    return uploaded

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
                success, attempted = process_and_upload_candidate(clip, can_upload_yt, can_upload_yt2, can_upload_ig)
                if attempted:
                    upload_attempts += 1

                if success:
                    uploaded = True
                    print(f"Successfully processed and published clip {clip_id} from {hours}h window!")
                    break

                if upload_attempts >= MAX_UPLOAD_ATTEMPTS_PER_RUN:
                    print("Reached max upload attempts for this run. Exiting cleanly.")
                    return uploaded

            if uploaded:
                break

            print(f"No clean, valid clips found in {hours}h window. Expanding search window...")

    if not uploaded:
        print("No valid GTA VI clips found across all progressive fallback windows.")
    return uploaded


def main():
    schedule = load_mc_schedule()
    yt_count, yt2_count, ig_count = get_daily_counts()

    now_ist = ist_now()
    today_ist = now_ist.strftime("%Y-%m-%d")
    is_morning = now_ist.hour < 12
    is_night = now_ist.hour >= 18

    has_yt2 = bool(os.getenv("YT_REFRESH_TOKEN_2", "").strip())
    has_yt3 = bool(os.getenv("YT_REFRESH_TOKEN_3", "").strip())

    print(f"Today's current upload count: YouTube={yt_count}/{MAX_YT_PER_DAY}, YouTube#2={yt2_count}/{MAX_YT2_PER_DAY}, Instagram={ig_count}/{MAX_IG_PER_DAY}")
    print(f"Current IST time: {now_ist.strftime('%Y-%m-%d %H:%M')} | UPLOAD_TARGET={UPLOAD_TARGET}")

    yt3_resting = UPLOAD_TARGET == "all" and yt3_queue_is_resting(schedule)
    yt2_morning_covered = is_morning and today_ist in schedule.get("yt2_morning_scheduled_dates", [])

    can_upload_yt = yt_count < MAX_YT_PER_DAY
    can_upload_yt2 = has_yt2 and yt2_count < MAX_YT2_PER_DAY and is_night
    can_upload_ig = ig_count < MAX_IG_PER_DAY
    run_yt3 = False
    run_yt2_mc = False

    if UPLOAD_TARGET == "yt3_only":
        can_upload_yt = False
        can_upload_yt2 = False
        can_upload_ig = False
        run_yt3 = has_yt3
        print("UPLOAD_TARGET=yt3_only: forcing Channel #3 Minecraft batch; Channels #1/#2 and Instagram disabled.")
        if not has_yt3:
            print("YT_REFRESH_TOKEN_3 is not set. Nothing to do.")
            return
    elif UPLOAD_TARGET == "yt2_only":
        can_upload_yt = False
        can_upload_ig = False
        can_upload_yt2 = has_yt2 and yt2_count < MAX_YT2_PER_DAY and not is_morning
        run_yt2_mc = has_yt2 and yt2_count < MAX_YT2_PER_DAY and is_morning
        print("UPLOAD_TARGET=yt2_only: Channel #2 only (Minecraft in Morning IST, GTA 6 otherwise).")
        if not has_yt2:
            print("YT_REFRESH_TOKEN_2 is not set. Nothing to do.")
            return
    else:
        run_yt3 = has_yt3 and not yt3_resting
        if is_morning:
            if yt2_morning_covered:
                print("Channel #2 Morning Minecraft slot is already scheduled. Skipping Minecraft for Channel #2.")
            else:
                # Channel #3 may be resting on its queue, but Channel #2 still needs
                # its Morning Minecraft clip, so allow a Channel #2-only render.
                run_yt2_mc = has_yt2 and yt2_count < MAX_YT2_PER_DAY
        elif is_night:
            # Night batch reuses rendered clips for upcoming Channel #2 Morning slots.
            run_yt2_mc = has_yt2 and run_yt3
        if yt3_resting:
            print(f"Channel #3 queue is still scheduled through {schedule.get('yt3_last_scheduled_utc')}. Channel #3 rests this run.")
        elif not has_yt3:
            print("YT_REFRESH_TOKEN_3 is not set. Channel #3 Minecraft scheduling disabled.")

    gta_needed = can_upload_yt or can_upload_yt2 or can_upload_ig

    if not gta_needed and not run_yt3 and not run_yt2_mc:
        print("No eligible upload targets for this run. Exiting cleanly.")
        return

    mc_uploaded = False
    if run_yt3 or run_yt2_mc:
        clip_count = 4 if run_yt3 else 1
        print(f"\n=== Minecraft Batch: rendering {clip_count} clip(s) ===")
        try:
            mc_uploaded = run_minecraft_batch(
                schedule,
                clip_count=clip_count,
                upload_yt3=run_yt3,
                upload_yt2=run_yt2_mc,
                yt2_count=yt2_count,
            )
        except Exception as e:
            print(f"Minecraft batch failed: {e}")

    gta_uploaded = False
    if gta_needed:
        print("\n=== GTA 6 Twitch Pipeline ===")
        try:
            gta_uploaded = run_gta_pipeline(can_upload_yt, can_upload_yt2, can_upload_ig)
        except Exception as e:
            print(f"GTA pipeline failed: {e}")

    if not mc_uploaded and not gta_uploaded:
        print("No uploads completed this run.")


if __name__ == "__main__":
    main()
