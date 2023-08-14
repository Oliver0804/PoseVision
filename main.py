import os
import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import torch

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
    (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16), (4, 11), (1, 12),
    (16, 20), (16, 22), (16, 18),
    (15, 19), (15, 21), (15, 17)
]

# 將 PNG 資料夾中的所有圖片合併成一個 MP4 檔案
def pngs_to_mp4(png_folder, output_mp4, fps=30):
    images = [img for img in os.listdir(png_folder) if img.endswith(".png")]
    images.sort()  # 確保圖像是按正確的順序

    frame = cv2.imread(os.path.join(png_folder, images[0]))
    height, width, layers = frame.shape

    # 使用第一幅圖片的尺寸來設定輸出MP4的尺寸，並設定 fps 為 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(png_folder, image)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


# 處理影片 
def process_video(input_path, output_path):
    # 初始化姿勢估計物件
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

    # 讀取影片
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))# 影片尺寸
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))# 影片尺寸
    fps = int(cap.get(cv2.CAP_PROP_FPS))# 幀率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))# 總幀數
    scale_factor = 3

    # 創建影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * scale_factor, height * scale_factor))

    vectors_list = []
    prev_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width * scale_factor, height * scale_factor))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in results.pose_landmarks.landmark]
            for index, point in enumerate(landmarks):
                x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 10, (250, 255, 50), 2)
                cv2.putText(frame, str(index), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 200), 2)
            
            if prev_landmarks is not None:
                movement_vectors = np.array(landmarks) - np.array(prev_landmarks)
                vectors_list.append(movement_vectors)
                for point, vector in zip(landmarks, movement_vectors):
                    start_point = (int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0]))
                    end_point = (int((point[0] + vector[0]) * frame.shape[1]), int((point[1] + vector[1]) * frame.shape[0]))
                    vector_length = np.linalg.norm(vector) * 500
                    color = (0, 255, 0)  # 綠色
                    if vector_length > 10:
                        color = (0, 0, 255)  # 紅色
                    elif vector_length > 5:
                        color = (0, 165, 255)  # 橘色
                    cv2.arrowedLine(frame, start_point, end_point, color, 2)

            prev_landmarks = landmarks

            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = (int(landmarks[start_idx][0] * frame.shape[1]), int(landmarks[start_idx][1] * frame.shape[0]))
                    end_point = (int(landmarks[end_idx][0] * frame.shape[1]), int(landmarks[end_idx][1] * frame.shape[0]))
                    cv2.line(frame, start_point, end_point, (255, 255, 255), 2)

        cv2.imshow('Pose Estimation', frame)
        out.write(frame)
        time.sleep(0.05)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    torch.save(vectors_list, "vectors.pt")
    print(f"影像尺寸: {width}x{height}")
    print(f"總幀數: {total_frames}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='input', help='Input video path or PNG folder')
    parser.add_argument('--out', dest='output', help='Output video path', default=None)  # 預設為None
    args = parser.parse_args()

    input_path = args.input
    output_video = args.output

    # 如果使用者沒有提供output，根據input生成output名稱
    if not output_video:
        base_name, ext = os.path.splitext(input_path)
        output_video = base_name + "_output.mp4"

    # 檢查輸入是資料夾還是 MP4
    if os.path.isdir(input_path):
        temp_mp4 = "temp_video.mp4"
        pngs_to_mp4(input_path, temp_mp4)
        process_video(temp_mp4, output_video)
        os.remove(temp_mp4)  # 刪除暫時的 MP4 檔案
    else:
        process_video(input_path, output_video)
