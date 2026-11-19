import os
import re
import shutil
import subprocess
import tempfile
import cv2
import mediapipe as mp
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
    Run mediapipe face detection.
    Filter for faces on sides (center_x < 0.35 * width or center_x > 0.65 * width).
    If >= 3 frames detect side face, compute median (face_cx, face_cy)
    and return vertical 9:16 crop box (crop_x, crop_y, crop_w, crop_h).
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

    frame_indices = np.linspace(0, total_frames - 1, num=12, dtype=int)
    side_face_centers = []

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    cx = (bbox.xmin + bbox.width / 2.0) * width
                    cy = (bbox.ymin + bbox.height / 2.0) * height

                    # Side face condition
                    if cx < 0.35 * width or cx > 0.65 * width:
                        side_face_centers.append((cx, cy))
                        break

    cap.release()

    if len(side_face_centers) < 3:
        print(f"Streamer cam detection: Only {len(side_face_centers)}/12 side faces detected. No webcam box.")
        return None

    med_cx = float(np.median([c[0] for c in side_face_centers]))
    med_cy = float(np.median([c[1] for c in side_face_centers]))
    print(f"Streamer cam detected! Detected {len(side_face_centers)} side faces. Median center: ({med_cx:.1f}, {med_cy:.1f})")

    # 9:16 crop box
    target_aspect = 9.0 / 16.0
    crop_h = height
    crop_w = int(crop_h * target_aspect)

    if crop_w > width:
        crop_w = width
        crop_h = int(crop_w / target_aspect)

    crop_x = int(med_cx - crop_w / 2.0)
    crop_y = int(med_cy - crop_h / 2.0)

    # Clamp bounds cleanly
    crop_x = max(0, min(crop_x, width - crop_w))
    crop_y = max(0, min(crop_y, height - crop_h))

    # Ensure even dimensions for video encoding
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
    Fills in non-speech gameplay segments between active segments.
    Keeps total kept duration >= 10s. If trimmed < 10s, keeps original untrimmed duration.
    Remaps word timestamps to match jump-cut timeline.
    Returns: timeline_segments (list of dicts with start, end, is_speech), remapped_words
    """
    if not all_words:
        # No speech words detected, single gameplay segment
        timeline = [{"start": 0.0, "end": original_duration, "is_speech": False}]
        return timeline, []

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

    # 4. If total speech duration is < 10s, keep original untrimmed duration
    if total_kept_speech < 10.0:
        print(f"Total speech duration ({total_kept_speech:.2f}s) < 10s. Keeping original full duration.")
        # Mark entire clip timeline, tagging active speaking intervals
        timeline = []
        curr_t = 0.0
        for seg in merged_speech_segments:
            if seg["start"] > curr_t:
                timeline.append({"start": curr_t, "end": seg["start"], "is_speech": False})
            timeline.append({"start": seg["start"], "end": seg["end"], "is_speech": True})
            curr_t = seg["end"]
        if curr_t < original_duration:
            timeline.append({"start": curr_t, "end": original_duration, "is_speech": False})

        # No jump cut time shift: words keep original timestamps
        remapped_words = all_words
        return timeline, remapped_words

    # Build jump-cut timeline (only kept speech segments are retained)
    timeline = []
    remapped_words = []
    accumulated_time = 0.0

    for seg in merged_speech_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        timeline.append({"start": seg_start, "end": seg_end, "is_speech": True})

        # Remap words within this segment
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

        accumulated_time += (seg_end - seg_start)

    print(f"Jump-cut silence removal applied: {original_duration:.2f}s -> {accumulated_time:.2f}s")
    return timeline, remapped_words


def write_ass_subtitles(remapped_words, ass_path):
    # MarginV=580 and Fontsize=76 to place subtitles cleanly above YouTube Shorts UI
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,Arial,76,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,40,40,580,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = []
    for w in remapped_words:
        w_text = w["word"].strip().upper() if isinstance(w, dict) else w.word.strip().upper()
        w_start = w["start"] if isinstance(w, dict) else w.start
        w_end = w["end"] if isinstance(w, dict) else w.end
        lines.append(f"Dialogue: 0,{format_ass_time(w_start)},{format_ass_time(w_end)},Default,,0,0,0,,{w_text}")

    Path(ass_path).write_text(header + "\n".join(lines), encoding="utf-8")


def render_dynamic_vertical(raw_mp4, ass_path, out_mp4, timeline_segments, cam_box):
    """
    Renders segments dynamically:
    - If cam_box and is_speech: cropped to streamer's webcam box (1080x1920)
    - Otherwise (gameplay): center crop (1080x1920)
    Concatenates segments and burns in subtitles.
    """
    temp_dir = tempfile.mkdtemp(prefix="render_dynamic_")
    try:
        segment_files = []
        for i, seg in enumerate(timeline_segments):
            seg_start = seg["start"]
            seg_dur = seg["end"] - seg["start"]
            if seg_dur <= 0.05:
                continue

            seg_file = os.path.join(temp_dir, f"seg_{i:03d}.mp4")

            if cam_box and seg["is_speech"]:
                cx, cy, cw, ch = cam_box
                vf = f"crop={cw}:{ch}:{cx}:{cy},scale=1080:1920"
            else:
                vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{seg_start:.3f}",
                "-i", str(raw_mp4),
                "-t", f"{seg_dur:.3f}",
                "-vf", vf,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-avoid_negative_ts", "make_zero",
                seg_file
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            segment_files.append(seg_file)

        if not segment_files:
            raise RuntimeError("No valid segments rendered.")

        concat_list = os.path.join(temp_dir, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for s in segment_files:
                f.write(f"file '{Path(s).as_posix()}'\n")

        # Concatenate and burn ASS subtitles
        concat_mp4 = os.path.join(temp_dir, "concat.mp4")
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            concat_mp4
        ]
        subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Burn subtitles
        escaped_ass = Path(ass_path).as_posix().replace(":", r"\:")
        burn_cmd = [
            "ffmpeg", "-y",
            "-i", concat_mp4,
            "-vf", f"ass={escaped_ass}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(out_mp4)
        ]
        subprocess.run(burn_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

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
    print(f"\nEvaluating candidate clip ({clip.get('view_count', 0)} views): {clip_id} - '{title}'")

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

        # 3. Detect streamer webcam position
        cam_box = detect_streamer_cam(raw_mp4)

        # 4. Jump-cut silence removal & timeline remapping
        orig_dur = get_video_duration(raw_mp4)
        timeline_segments, remapped_words = build_jumpcut_timeline(words, orig_dur)

        # 5. Generate high subtitles (MarginV=580, Fontsize=76)
        write_ass_subtitles(remapped_words, ass_path)

        # 6. Render dynamic vertical video with speaker-to-gameplay switching
        render_dynamic_vertical(raw_mp4, ass_path, out_mp4, timeline_segments, cam_box)

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
