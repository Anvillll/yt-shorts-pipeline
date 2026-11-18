import os
import subprocess
import requests
from pathlib import Path
from faster_whisper import WhisperModel
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

PROCESSED_FILE = Path("processed.txt")
PROCESSED = set(PROCESSED_FILE.read_text().splitlines()) if PROCESSED_FILE.exists() else set()


def get_twitch_token():
    r = requests.post(
        "https://id.twitch.tv/oauth2/token",
        data={
            "client_id": os.environ["TWITCH_CLIENT_ID"],
            "client_secret": os.environ["TWITCH_CLIENT_SECRET"],
            "grant_type": "client_credentials",
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def get_gta_vi_game_id(token):
    headers = {
        "Client-ID": os.environ["TWITCH_CLIENT_ID"],
        "Authorization": f"Bearer {token}",
    }
    r = requests.get(
        "https://api.twitch.tv/helix/games",
        params={"name": "Grand Theft Auto VI"},
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise ValueError("Could not find Twitch game ID for 'Grand Theft Auto VI'.")
    game_id = data[0]["id"]
    print(f"Found Grand Theft Auto VI game_id: {game_id}")
    return game_id


def get_top_gta_vi_clips(token, game_id, limit=20):
    headers = {
        "Client-ID": os.environ["TWITCH_CLIENT_ID"],
        "Authorization": f"Bearer {token}",
    }
    r = requests.get(
        "https://api.twitch.tv/helix/clips",
        params={"game_id": game_id, "first": limit},
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    clips = r.json().get("data", [])
    # Sort clips descending by view_count
    clips.sort(key=lambda c: c.get("view_count", 0), reverse=True)
    return clips


def download_clip(clip_url, output_path):
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        clip_url,
    ]
    subprocess.run(cmd, check=True)


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


def main():
    token = get_twitch_token()
    game_id = get_gta_vi_game_id(token)
    clips = get_top_gta_vi_clips(token, game_id)

    # Filter clips: 10 to 60 seconds duration, not already processed
    candidate_clip = None
    for clip in clips:
        clip_id = clip["id"]
        duration = clip.get("duration", 0)
        if clip_id in PROCESSED:
            continue
        if 10 <= duration <= 60:
            candidate_clip = clip
            break

    if not candidate_clip:
        print("No unprocessed Grand Theft Auto VI clips found.")
        return

    clip_id = candidate_clip["id"]
    clip_url = candidate_clip["url"]
    clip_title = candidate_clip.get("title", "GTA VI Clip")
    print(f"Selected clip: {clip_id} - '{clip_title}' (Views: {candidate_clip.get('view_count')}, Duration: {candidate_clip.get('duration')}s)")

    raw_mp4 = Path("raw_clip.mp4")
    ass_path = Path("subtitles.ass")
    out_mp4 = Path("final_short.mp4")

    try:
        print("Downloading clip...")
        download_clip(clip_url, raw_mp4)

        print("Generating subtitles with Whisper...")
        generate_ass(raw_mp4, ass_path)

        print("Rendering vertical video with subtitles...")
        render_vertical(raw_mp4, ass_path, out_mp4)

        print("Uploading to YouTube...")
        upload_youtube(out_mp4, clip_title)

        # Mark clip as processed
        with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
            f.write(f"{clip_id}\n")
        print(f"Clip {clip_id} added to {PROCESSED_FILE}.")

    finally:
        # Cleanup temporary media files
        for p in (raw_mp4, ass_path, out_mp4):
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
