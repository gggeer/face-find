import streamlit as st
import tempfile
import cv2
import time
import face_detection
import numpy as np


def extract_faces_from_image(im, bboxes):
    extracted_faces = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Adaptive Histogram Equalization

    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cropped_face = im[y0:y1, x0:x1]
        cropped_face = cv2.resize(cropped_face, (cropped_face.shape[1] * 2, cropped_face.shape[0] * 2))

        for channel in range(cropped_face.shape[2]):
            cropped_face[:, :, channel] = clahe.apply(cropped_face[:, :, channel])

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        cropped_face = cv2.filter2D(cropped_face, -1, kernel)
        extracted_faces.append(cropped_face)

    return extracted_faces


def process_video(video_file):
    video_path = video_file.name
    vidcap = cv2.VideoCapture(video_path)
    detector = face_detection.build_detector("DSFDDetector", max_resolution=1080)

    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    all_faces = []
    frame_count = 0
    while True:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * fps)

        success, im = vidcap.read()
        if not success:
            break

        dets = detector.detect(im[:, :, ::-1])[:, :4]
        faces_in_frame = extract_faces_from_image(im, dets)
        all_faces.extend(faces_in_frame)
        frame_count += 1

    return all_faces


# Streamlit UI
st.title("Face Extraction from Video")
st.write("Upload an MP4 video to extract faces from it.")

uploaded_video = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    st.video(tfile.name)

    if st.button("Extract Faces"):
        extracted_faces = process_video(tfile)

        st.write("Extracted Faces:")
        cols = st.columns (3)  # Create 3 columns for the gallery

        for idx, face_img in enumerate(extracted_faces):
            with cols[idx % 3]:  # Cycle through columns
                st.image(face_img, channels="BGR", caption=f"Face #{idx + 1}")
