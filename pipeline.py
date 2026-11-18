import os, subprocess, requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from faster_whisper import WhisperModel
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

PROCESSED_FILE = Path("processed.txt")
PROCESSED = set(PROCESSED_FILE.read_text().splitlines()) if PROCESSED_FILE.exists() else set()
MAX_UPLOADS_PER_DAY = 6  # Matches YouTube's 10,000 daily API quota (1,600 units/upload)

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
    r = requests.post("https://twitch.tv", data={
        "client_id": os.environ["TWITCH_CLIENT_ID"],
        "client_secret": os.environ["TWITCH_CLIENT_SECRET"],
        "grant_type": "client_credentials",
    }, timeout=15)
    r.raise_for_status()
    return r.json()["access_token"]

def get_gta6_game_id(token):
    headers = {"Client-ID": os.environ["TWITCH_CLIENT_ID"], "Authorization": f"Bearer {token}"}
    r = requests.get("https://twitch.tv",
                     headers=headers, params={"name": "Grand Theft Auto VI"}, timeout=15)
    r.raise_for_status()
    data = r.json().get("data", [])
    return data[0]["id"] if data else None

def fetch_todays_top_clip(token, game_id):
    headers = {"Client-ID": os.environ["TWITCH_CLIENT_ID"], "Authorization": f"Bearer {token}"}
    started_at = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://twitch.tv", headers=headers, params={
        "game_id": game_id,
        "started_at": started_at,
        "first": 20,
    }, timeout=15)
    r.raise_for_status()
    clips = sorted(r.json().get("data", []), key=lambda c: c.get("view_count", 0), reverse=True)
    for clip in clips:
        if clip["id"] not in PROCESSED and 10 <= clip["duration"] <= 60:
            return clip
    return None

def format_ass_time(sec):
    h, m = int(sec // 3600), int((sec % 3600) // 60)
    s, cs = int(sec % 60), int((sec % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def generate_ass(video_path, ass_path):
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(video_path), word_timestamps=True)
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
        token_uri="https://googleapis.com",
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

def main():
    if reached_daily_limit():
        print("Reached 6 uploads today (YouTube API daily quota limit). Skipping until tomorrow.")
        return

    token = get_twitch_token()
    game_id = get_gta6_game_id(token)
    if not game_id:
        print("Grand Theft Auto VI category not found on Twitch.")
        return

    clip = fetch_todays_top_clip(token, game_id)
    if not clip:
        print("No new GTA VI clips found in the last 24 hours.")
        return

    clip_id = clip["id"]
    print(f"Processing today's top clip ({clip['view_count']} views): {clip_id} - {clip['title']}")
    raw_mp4 = Path(f"{clip_id}_raw.mp4")
    ass_path = Path(f"{clip_id}.ass")
    out_mp4 = Path(f"{clip_id}_short.mp4")

    subprocess.run(["yt-dlp", "-f", "mp4", "-o", str(raw_mp4), clip["url"]], check=True)
    generate_ass(raw_mp4, ass_path)
    render_vertical(raw_mp4, ass_path, out_mp4)
    upload_youtube(out_mp4, clip["title"])

    with PROCESSED_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{clip_id}\n")
    increment_daily_count()

if __name__ == "__main__":
    main()
