import cv2
import mediapipe as mp
import numpy as np
import time
import argparse

def process_video(input_path, output_path):
    # 初始化姿勢估計物件
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # 讀取影片
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 儲存前一幀的關鍵點位置
    prev_landmarks = None

    # 視窗放大倍數
    scale_factor = 3

    # 創建影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * scale_factor, height * scale_factor))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 將視窗放大
        frame = cv2.resize(frame, (width * scale_factor, height * scale_factor))

        # 將圖像轉換為RGB顏色空間
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 執行姿勢估計
        results = pose.process(frame_rgb)

        # 檢測到人體姿勢的關鍵點
        if results.pose_landmarks:
            # 取得當前幀的關鍵點位置
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y))

            # 繪製關鍵點
            for point in landmarks:
                x = int(point[0] * frame.shape[1])
                y = int(point[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # 計算移動向量
            if prev_landmarks is not None:
                movement_vectors = np.array(landmarks) - np.array(prev_landmarks)

                # 繪製移動向量
                for point, vector in zip(landmarks, movement_vectors):
                    start_point = (int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0]))
                    end_point = (int((point[0] + vector[0]) * frame.shape[1]), int((point[1] + vector[1]) * frame.shape[0]))

                    # 計算向量的絕對值
                    vector_length = np.linalg.norm(vector)
                    scale_vector_length = vector_length * 500
                    # 根據向量大小設定箭頭顏色
                    if scale_vector_length > 10:
                        print("Red")
                        color = (0, 0, 255)  # 紅色
                    elif scale_vector_length > 5:
                        print("Orange")
                        color = (0, 165, 255)  # 橘色
                    else:
                        print("Green")
                        color = (0, 255, 0)  # 綠色

                    cv2.arrowedLine(frame, start_point, end_point, color, 2)

                    # 打印移動向量數值
                    print(f"Keypoint: {point}, Vector: {vector}")

            # 儲存當前幀的關鍵點位置，供下一幀使用
            prev_landmarks = landmarks

        # 顯示結果
        cv2.imshow('Pose Estimation', frame)
        out.write(frame)
        time.sleep(0.05)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 解析命令行參數
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='input', help='Input video path')
    parser.add_argument('--out', dest='output', help='Output video path')
    args = parser.parse_args()

    # 使用命令行參數執行影片處理
    input_video = args.input
    output_video = args.output
    process_video(input_video, output_video)
