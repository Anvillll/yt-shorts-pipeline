import os
import re
import shutil
import subprocess
import tempfile
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from faster_whisper import WhisperModel
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from nudenet import NudeDetector

PROCESSED_FILE = Path("processed.txt")
PROCESSED = set(PROCESSED_FILE.read_text().splitlines()) if PROCESSED_FILE.exists() else set()
MAX_UPLOADS_PER_DAY = 6  # Matches YouTube's 10,000 daily API quota (1,600 units/upload)
FALLBACK_HOURS = [24, 48, 72, 168, 720]  # 24h -> 48h -> 3 days -> 7 days -> 30 days

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


def reached_daily_limit():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count_file = Path("daily_count.txt")
    if count_file.exists():
        content = count_file.read_text().strip()
        if ":" in content:
            saved_date, count = content.split(":")
            if saved_date == today_str and int(count) >= MAX_UPLOADS_PER_DAY:
                return True
    return False


def increment_daily_count():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count_file = Path("daily_count.txt")
    count = 0
    if count_file.exists():
        content = count_file.read_text().strip()
        if ":" in content:
            saved_date, saved_count = content.split(":")
            if saved_date == today_str:
                count = int(saved_count)
    count_file.write_text(f"{today_str}:{count + 1}", encoding="utf-8")


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


def generate_ass(video_path, ass_path):
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(video_path), word_timestamps=True)

    # Verify detected language is English with probability >= 0.75
    if info.language != "en" or info.language_probability < 0.75:
        print(f"Whisper rejected audio: detected language='{info.language}' with prob={info.language_probability:.2f}")
        return False

    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,Arial,72,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,40,40,320,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = []
    for seg in segments:
        for w in seg.words or []:
            lines.append(f"Dialogue: 0,{format_ass_time(w.start)},{format_ass_time(w.end)},Default,,0,0,0,,{w.word.strip().upper()}")
    Path(ass_path).write_text(header + "\n".join(lines), encoding="utf-8")
    return True


def render_vertical(raw_mp4, ass_path, out_mp4):
    vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,ass={ass_path}"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw_mp4), "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out_mp4)
    ], check=True)


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
            "title": f"{title[:85]} #Shorts",
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


def process_and_upload_candidate(clip):
    clip_id = clip["id"]
    title = clip.get("title", "")
    print(f"\nEvaluating candidate clip ({clip.get('view_count', 0)} views): {clip_id} - '{title}'")

    raw_mp4 = Path(f"{clip_id}_raw.mp4")
    ass_path = Path(f"{clip_id}.ass")
    out_mp4 = Path(f"{clip_id}_short.mp4")

    try:
        subprocess.run(["yt-dlp", "-f", "mp4", "-o", str(raw_mp4), clip["url"]], check=True)

        # Strict Nudity & Adult-Content Guard
        if contains_nsfw(raw_mp4):
            print(f"WARNING: Clip {clip_id} flagged for NSFW/nudity content! Skipping.")
            mark_processed(clip_id)
            return False

        # Audio transcription & English check
        english_audio = generate_ass(raw_mp4, ass_path)
        if not english_audio:
            print(f"Clip {clip_id} audio was not verified English. Skipping.")
            mark_processed(clip_id)
            return False

        render_vertical(raw_mp4, ass_path, out_mp4)
        upload_youtube(out_mp4, title)

        mark_processed(clip_id)
        increment_daily_count()
        return True

    except Exception as e:
        print(f"Error processing clip {clip_id}: {e}")
        mark_processed(clip_id)
        return False

    finally:
        for p in (raw_mp4, ass_path, out_mp4):
            if p.exists():
                p.unlink()


def main():
    if reached_daily_limit():
        print("Reached 6 uploads today (YouTube API daily quota limit). Skipping until tomorrow.")
        return

    token = get_twitch_token()
    game_id = get_gta6_game_id(token)
    if not game_id:
        print("Grand Theft Auto VI category not found on Twitch.")
        return

    # Check clips across progressive time windows
    uploaded = False
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

            # Attempt full download, NSFW check, Whisper check, render, and upload
            if process_and_upload_candidate(clip):
                uploaded = True
                print(f"Successfully processed and uploaded clip {clip_id} from {hours}h window!")
                break

        if uploaded:
            break

        print(f"No clean, valid clips found in {hours}h window. Expanding search window...")

    if not uploaded:
        print("No valid GTA VI clips found across all progressive fallback windows.")


if __name__ == "__main__":
    main()
