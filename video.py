import cv2
import os

# Function to adjust brightness
def adjust_brightness(frame, value=30):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    frame_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return frame_bright

def process_single_video(video_path, output_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    ext = os.path.splitext(video_path)[1]
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare video writers with modified names
    flipped_video = os.path.join(output_dir, f'{video_name}_flipped{ext}')
    bright_video = os.path.join(output_dir, f'{video_name}_light{ext}')
    dark_video = os.path.join(output_dir, f'{video_name}_dark{ext}')

    out_flip = cv2.VideoWriter(flipped_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_bright = cv2.VideoWriter(bright_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_dark = cv2.VideoWriter(dark_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame
        flipped_frame = cv2.flip(frame, 1)

        # Brighten the frame
        bright_frame = adjust_brightness(frame, value=40)

        # Darken the frame
        dark_frame = adjust_brightness(frame, value=-40)

        # Write transformed frames into respective videos
        out_flip.write(flipped_frame)
        out_bright.write(bright_frame)
        out_dark.write(dark_frame)

    cap.release()
    out_flip.release()
    out_bright.release()
    out_dark.release()

    cv2.destroyAllWindows()

# Process all videos in each subfolder
def process_all_videos_in_folders(root_dir):
    for subfolder in [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]:
        subfolder_path = os.path.join(root_dir, subfolder)
        for video_file in os.listdir(subfolder_path):
            if video_file.endswith(('.mp4', '.mov', '.MOV')):
                video_path = os.path.join(subfolder_path, video_file)
                process_single_video(video_path, subfolder_path)

# Root directory
root_dir = "Greetings"

process_all_videos_in_folders(root_dir)