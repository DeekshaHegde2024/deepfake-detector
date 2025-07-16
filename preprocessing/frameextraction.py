import os
import cv2
from mtcnn import MTCNN
import re

def frame_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_top_confident_frames(
    video_path, output_folder,
    video_type, video_number,
    num_frames=3,
    confidence_threshold=0.90,
    sharpness_threshold=50,
    sampling_interval=3,
    resize_dim=(299, 299)
):
    cap      = cv2.VideoCapture(video_path)
    detector = MTCNN()
    scored   = []
    idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sampling_interval == 0:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = frame_sharpness(gray)

            dets = detector.detect_faces(rgb)
            if dets:
                best_conf = max(face["confidence"] for face in dets)
                if best_conf >= confidence_threshold and sharp >= sharpness_threshold:
                    scored.append((best_conf * sharp, frame.copy()))
                    if len(scored) == num_frames:
                        break
            else:
                print(f"Frame {idx}: Sharpness={sharp:.2f}, Best_conf=None")
        idx += 1

    cap.release()
    scored.sort(reverse=True, key=lambda x: x[0])
    chosen = scored[:num_frames]
    saved = []

    for i, (_, frm) in enumerate(chosen, start=1):
        out = cv2.resize(frm, resize_dim, interpolation=cv2.INTER_AREA)
        name = f"{video_type}_{video_number:03d}_{i:02d}.jpg"
        path = os.path.join(output_folder, name)
        cv2.imwrite(path, out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved.append(path)

    print(f"[{os.path.basename(video_path)}] Saved {len(chosen)} frames (target {num_frames}).")
    return len(saved)


def process_video_folder(
    input_folder, output_folder,
    video_type="Face2Face",
    num_frames=3,
    confidence_threshold=0.90,
    sharpness_threshold=50,
    sampling_interval=3,
    resize_dim=(299, 299),
    max_images_per_batch=3000
):
    os.makedirs(output_folder, exist_ok=True)

    # Count already saved images
    existing = [
        f for f in os.listdir(output_folder)
        if f.startswith(video_type + "_") and f.endswith(".jpg")
    ]
    total_saved = len(existing)
    videos_done = total_saved // num_frames

    print(f"Resuming from video {videos_done + 1}, {total_saved} images already saved.")

    videos = sorted(f for f in os.listdir(input_folder) if f.lower().endswith(".mp4"))

    for vid in videos[videos_done:]:
        if total_saved >= (videos_done * num_frames) + max_images_per_batch:
            print(f"Reached batch limit of {max_images_per_batch} new images. Stopping.")
            break

        # Extract video number from filename
        match = re.search(r"(\d+)", vid)
        if match:
            video_number = int(match.group(1))
        else:
            print(f"Warning: Could not extract number from filename '{vid}'. Skipping.")
            continue

        saved = extract_top_confident_frames(
            os.path.join(input_folder, vid),
            output_folder,
            video_type, video_number,
            num_frames=num_frames,
            confidence_threshold=confidence_threshold,
            sharpness_threshold=sharpness_threshold,
            sampling_interval=sampling_interval,
            resize_dim=resize_dim
        )
        total_saved += saved

# --- Example usage ---
input_folder  = r"D:/Deepfake_dataset1/Fake_F2F"
output_folder = r"D:/Deepfake_dataset1/Fake_Frames_F2F"

process_video_folder(
    input_folder, output_folder,
    video_type="Face2Face",
    num_frames=3,
    confidence_threshold=0.90,
    sharpness_threshold=50,
    sampling_interval=3,
    resize_dim=(299, 299),
    max_images_per_batch=3000
)
