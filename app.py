import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model('weights/sign_language.h5')

# Mediapipe model for pose and landmarks detection
mp_holistic = mp.solutions.holistic

# Define the label map
label_map = {
    0: "48. Hello",
    1: "49. How are you",
    2: "50. Alright",
    3: "51. Good morning",
    4: "52. Good afternoon",
    5: "53. Good evening",
    6: "54. Good night",
    7: "55. Thank you",
    8: "56. Pleased"
}

# Helper functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def convert_video_to_pose_embedded_np_array(video_path, frames_to_extract=45):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    np_array = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        if total_frames >= frames_to_extract:
            frame_indices = np.linspace(0, total_frames - 1, frames_to_extract, dtype=int)
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                np_array.append(keypoints)
        else:
            key_points_shape = None
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                if key_points_shape is None:
                    key_points_shape = keypoints.shape
                np_array.append(keypoints)

            for _ in range(frames_to_extract - total_frames):
                np_array.append(np.zeros(shape=key_points_shape))

    cap.release()

    # Padding if video had fewer frames
    while len(np_array) < frames_to_extract:
        np_array.append(np.zeros_like(np_array[0]))

    return np.array(np_array)

def predict_on_video(model, video_path, frames_to_extract=15):
    input_data = convert_video_to_pose_embedded_np_array(video_path, frames_to_extract)
    input_data = np.expand_dims(input_data, axis=0)
    predictions = model.predict(input_data)[0]
    top_index = np.argmax(predictions)
    top_conf = predictions[top_index]
    return top_index, top_conf, predictions

# Streamlit app
st.title("Indian Sign Language Detection")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(uploaded_video)

    if st.button("Predict"):
        st.info("Processing and analyzing video...")

        try:
            pred_class, confidence, all_preds = predict_on_video(model, video_path)
            if pred_class in label_map:
                st.success(f"✅ Predicted Phrase: **{label_map[pred_class]}**")
                # st.write(f"📊 Confidence: **{confidence:.2f}**")
            else:
                st.warning("Prediction does not match known class labels.")
        except Exception as e:
            st.error(f"Error occurred: {e}")
