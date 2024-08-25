import os
import cv2
import mediapipe as mp
import numpy as np
import argparse
import pandas as pd


from googletrans import Translator
from colorama import init, Fore, Style

# 閥值設定
THRESHOLD = 0.2
# 適合深度學習的資料結構
vectors_data = []

# 初始化 colorama
init(autoreset=True)

translator = Translator()


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
# TODO: Load the CSV file
# 需修改去使用--in名稱從name欄位找到對應的orth和translation
# 並且將orth翻譯成英文和中文
# 並且將orth, translation, translation_en, translation_zh_tw印出來


def load_csv_data(csv_file, folder_name):
    # 讀取 CSV 檔案
    df = pd.read_csv(csv_file, delimiter='|')
    # 檢查 CSV 檔案的欄位名稱
    print(Fore.YELLOW + "CSV Columns:", df.columns)
    
    # 根據 folder_name 找到對應的 row
    try:
        row = df[df['name'] == folder_name].iloc[0]
    except IndexError:
        raise ValueError(f"Folder name '{folder_name}' not found in CSV file.")
    
    # 獲取 orth 和 translation
    orth = row['orth']
    translation = row['translation']
    
    return orth, translation

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

def process_video(input_path, output_path, csv_file, folder_name):
        # 初始化 frame_number
    frame_number = 0
    orth, translation = load_csv_data(csv_file, folder_name)
    pose = mp_pose.Pose(min_detection_confidence=0.5)
    hands = mp_hands.Hands(min_detection_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height * 2))

    last_frame_landmarks = {}
    vectors_list = []
    # 翻譯文字
    translation_en = translator.translate(orth, dest='en').text
    translation_zh_tw = translator.translate(orth, dest='zh-tw').text

    # 打印帶有色彩的翻譯文字
    print(Fore.GREEN + f"Original: {orth}")
    print(Fore.CYAN + f"Translation: {translation}")
    print(Fore.YELLOW + f"Translation (EN): {translation_en}")
    print(Fore.MAGENTA + f"Translation (ZH-TW): {translation_zh_tw}")
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
                vector_magnitude = np.linalg.norm(vector)
                
                if np.all(np.abs(vector) <= THRESHOLD):  # Check if all components of the vector are within the threshold
                    vectors_list.append((key, vector))
                    start_point = (int(prev_value[0] * width), int(prev_value[1] * height))
                    end_point = (int(value[0] * width), int(value[1] * height))
                    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)  # Red line for vector
                    
                    # 打印向量資訊
                    print(f"Frame: {frame_number}, Landmark: {key}, Vector: {vector}")
                    
                    # 保存向量資訊到資料結構
                    vectors_data.append({
                        'frame': frame_number,
                        'landmark': key,
                        'vector': vector.tolist()
                    })

        # 更新 last_frame_landmarks
        last_frame_landmarks = current_frame_landmarks

        # 更新 frame_number
        frame_number += 1


        # Resize frame to double the size
        frame_resized = cv2.resize(frame, (width * 2, height * 2))

        last_frame_landmarks = current_frame_landmarks
        out.write(frame_resized)
        cv2.imshow('Video Feed', frame_resized)
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
    parser.add_argument('--csv', dest='csv_file', help='Path to the CSV file')
    args = parser.parse_args()
    input_path = args.input
    output_video = args.output or os.path.splitext(input_path)[0] + "_output.mp4"
    # 更清楚地打印每個參數
    print(Fore.YELLOW + "Parsed Arguments:")
    print(Fore.YELLOW + f"  Input Path: {args.input}")
    print(Fore.YELLOW + f"  Output Path: {args.output}")
    print(Fore.YELLOW + f"  CSV File: {args.csv_file}")
    folder_name=input_path.split('/')[-1]
    print(Fore.YELLOW + f"  Folder Name: {folder_name}")


    if os.path.isdir(input_path):
        temp_mp4 = "temp_video.mp4"
        pngs_to_mp4(input_path, temp_mp4)
        vectors = process_video(temp_mp4, output_video, args.csv_file,folder_name)
        os.remove(temp_mp4)
    else:
        vectors = process_video(input_path, output_video, args.csv_file, folder_name)
    
    print(Fore.GREEN + f"Total vectors captured: {len(vectors)}")