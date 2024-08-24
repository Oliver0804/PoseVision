import os
import cv2
import mediapipe as mp
import numpy as np
import argparse

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# 将PNG图像合并成一个MP4文件
def pngs_to_mp4(png_folder, output_mp4, fps=30):
    images = [img for img in os.listdir(png_folder) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(png_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    for image in images:
        frame = cv2.imread(os.path.join(png_folder, image))
        out.write(frame)
    out.release()

# 处理视频
# 处理视频函数
def process_video(input_path, output_path):
    pose = mp_pose.Pose(min_detection_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_frame_landmarks = {}
    vectors_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        current_frame_landmarks = {}

        # Process pose landmarks
        if results_pose.pose_landmarks:
            for idx, lm in enumerate(results_pose.pose_landmarks.landmark):
                current_frame_landmarks[f'pose_{idx}'] = (lm.x, lm.y)
                cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 2, (0, 255, 0), -1)

        # Process hand landmarks
        if results_hands.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                for idx, lm in enumerate(hand_landmarks.landmark):
                    current_frame_landmarks[f'hand_{hand_index}_{idx}'] = (lm.x, lm.y)
                    cv2.circle(frame, (int(lm.x * width), int(lm.y * height)), 2, (0, 255, 0), -1)

        # Calculate vectors only for landmarks that appeared in the last frame
        for key, value in current_frame_landmarks.items():
            if key in last_frame_landmarks:
                prev_value = last_frame_landmarks[key]
                vector = np.array(value) - np.array(prev_value)
                if np.linalg.norm(vector) > 0:  # Optional: check if movement is significant
                    vectors_list.append((key, vector))
                    start_point = (int(prev_value[0] * width), int(prev_value[1] * height))
                    end_point = (int(value[0] * width), int(value[1] * height))
                    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)  # Red line for vector

        last_frame_landmarks = current_frame_landmarks

        out.write(frame)
        cv2.imshow('Video Feed', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return vectors_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='input', help='Input video path or PNG folder')
    parser.add_argument('--out', dest='output', help='Output video path', default=None)
    args = parser.parse_args()

    input_path = args.input
    output_video = args.output or os.path.splitext(input_path)[0] + "_output.mp4"

    if os.path.isdir(input_path):
        temp_mp4 = "temp_video.mp4"
        pngs_to_mp4(input_path, temp_mp4)
        vectors = process_video(temp_mp4, output_video)
        os.remove(temp_mp4)
    else:
        vectors = process_video(input_path, output_video)
    print(f"Total vectors captured: {len(vectors)}")
