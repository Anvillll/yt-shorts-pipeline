import os
import re
import shutil
import subprocess
import tempfile
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
    Sample 12 evenly spaced frames using OpenCV.
    Detect faces using OpenCV built-in Haar Cascade classifier.
    Filter for faces on sides (face_cx < 0.35 * width or face_cx > 0.65 * width).
    If >= 3 frames detect side face, compute median (face_cx, face_cy, face_w, face_h)
    and return crop box (crop_x, crop_y, crop_w, crop_h) with 1080:680 (~1.59:1)
    aspect ratio centered around streamer's median face coordinates (padded to show
    head, shoulders, headset, clamped inside frame bounds).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Warning: Could not load Haar cascade from {cascade_path}")
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, num=12, dtype=int)
    side_faces = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(detected) > 0:
            for (x, y, w, h) in detected:
                cx = x + w / 2.0
                cy = y + h / 2.0
                # Side face condition
                if cx < 0.35 * width or cx > 0.65 * width:
                    side_faces.append((cx, cy, float(w), float(h)))
                    break

    cap.release()

    if len(side_faces) < 3:
        print(f"Streamer cam detection: Only {len(side_faces)}/12 side faces detected. No webcam box.")
        return None

    med_cx = float(np.median([f[0] for f in side_faces]))
    med_cy = float(np.median([f[1] for f in side_faces]))
    med_bw = float(np.median([f[2] for f in side_faces]))
    med_bh = float(np.median([f[3] for f in side_faces]))
    print(f"Streamer cam detected! Detected {len(side_faces)} side faces. Median center: ({med_cx:.1f}, {med_cy:.1f}), size: ({med_bw:.1f}, {med_bh:.1f})")

    # Target aspect ratio: 1080 / 680 (~1.5882)
    target_aspect = 1080.0 / 680.0

    # Generously pad around face to include head, headset, and shoulders (e.g. 3.2x face height)
    crop_h = int(med_bh * 3.2)
    crop_w = int(crop_h * target_aspect)

    # Make sure crop_h and crop_w are within bounds, but also sufficiently large
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

    # Shift center slightly down from face center to include shoulders
    target_cy = med_cy + med_bh * 0.15

    crop_x = int(med_cx - crop_w / 2.0)
    crop_y = int(target_cy - crop_h / 2.0)

    # Clamp bounds cleanly
    crop_x = max(0, min(crop_x, width - crop_w))
    crop_y = max(0, min(crop_y, height - crop_h))

    # Ensure even dimensions
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)
    crop_x = crop_x - (crop_x % 2)
    crop_y = crop_y - (crop_y % 2)

    return (crop_x, crop_y, crop_w, crop_h)


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


def transcribe_and_verify(video_path):
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(video_path), word_timestamps=True)

    if info.language != "en" or info.language_probability < 0.75:
        print(f"Whisper rejected audio: detected language='{info.language}' with prob={info.language_probability:.2f}")
        return None, None

    all_words = []
    for seg in segments:
        for w in seg.words or []:
            all_words.append(w)

    return all_words, info


def build_jumpcut_timeline(all_words, original_duration):
    """
    Groups consecutive words into active speech segments whenever silence gap <= 0.85s.
    Pads each active speech segment by 0.15s before start and 0.20s after end.
    Keeps total kept duration >= 10s. If trimmed < 10s, keeps original untrimmed duration.
    Ensures total kept duration is capped at 58.5s so video strictly qualifies as a Short.
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
    # Cap total accumulated duration at MAX_SHORT_DURATION (58.5s)
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


def build_hook_text(broadcaster_name, title):
    # Clean hashtags, links, and non-ASCII chars
    clean_title = re.sub(r"https?://\S+", "", title)
    clean_title = re.sub(r"#\S+", "", clean_title)
    clean_title = "".join(c for c in clean_title if 32 <= ord(c) < 127).strip()
    clean_title = re.sub(r"\s+", " ", clean_title)

    clean_broadcaster = "".join(c for c in broadcaster_name if 32 <= ord(c) < 127).strip()
    if clean_broadcaster:
        full_text = f"{clean_broadcaster.upper()}: {clean_title}"
    else:
        full_text = clean_title

    # Wrap to ~26 chars per line, max 2 lines
    words = full_text.split()
    line1 = []
    line2 = []
    len1 = 0
    for w in words:
        w_len = len(w) + (1 if line1 else 0)
        if len1 + w_len <= 26 and not line2:
            line1.append(w)
            len1 += w_len
        else:
            if len(" ".join(line2 + [w])) <= 26:
                line2.append(w)
            else:
                if not line2:
                    line2.append(w[:23] + "...")
                break

    text_line1 = " ".join(line1)
    text_line2 = " ".join(line2)

    if text_line2:
        return f"{text_line1}\\N{text_line2}"
    return text_line1


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
        # Filter non-ASCII characters for clean rendering
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
    broadcaster = clip.get("broadcaster_name", "")
    print(f"\nEvaluating candidate clip ({clip.get('view_count', 0)} views): {clip_id} - '{title}' by {broadcaster}")

    raw_mp4 = Path(f"{clip_id}_raw.mp4")
    ass_path = Path(f"{clip_id}.ass")
    out_mp4 = Path(f"{clip_id}_short.mp4")

    try:
        subprocess.run(["yt-dlp", "-f", "mp4", "-o", str(raw_mp4), clip["url"]], check=True)

        # 1. Strict Nudity & Adult-Content Guard
        if contains_nsfw(raw_mp4):
            print(f"WARNING: Clip {clip_id} flagged for NSFW/nudity content! Skipping.")
            mark_processed(clip_id)
            return False

        # 2. Audio transcription & English check
        words, info = transcribe_and_verify(raw_mp4)
        if words is None:
            print(f"Clip {clip_id} audio was not verified English. Skipping.")
            mark_processed(clip_id)
            return False

        # 3. Detect streamer webcam position (1080:680 box)
        cam_box = detect_streamer_cam(raw_mp4)

        # 4. Jump-cut silence removal & timeline remapping
        orig_dur = get_video_duration(raw_mp4)
        timeline_segments, remapped_words, total_dur = build_jumpcut_timeline(words, orig_dur)

        # 5. Build hook banner text and write ASS subtitles (Hook style at y=680, Default style at MarginV=420)
        hook_text = build_hook_text(broadcaster, title)
        write_ass_subtitles(remapped_words, ass_path, total_dur, hook_text, has_webcam=(cam_box is not None))

        # 6. Render stacked split-screen vertical video (top 1080x680 webcam, bottom 1080x1240 gameplay)
        render_split_vertical(raw_mp4, ass_path, out_mp4, timeline_segments, cam_box)

        # 7. Upload to YouTube
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
