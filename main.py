import os
import cv2
import mediapipe as mp
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt 
from googletrans import Translator
from colorama import init, Fore, Style

# 閥值設定
THRESHOLD = 0.1
# 適合深度學習的資料結構
vectors_data = []
# 畫面放大倍數
SCALE = 1
# 初始化 colorama
init(autoreset=True)

translator = Translator()

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


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
    if not images:
        return False
    frame = cv2.imread(os.path.join(png_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mp4, fourcc, fps, (width, height))
    for image in images:
        frame = cv2.imread(os.path.join(png_folder, image))
        out.write(frame)
    out.release()
    return True

def save_vectors_to_csv(vectors_data, input_path, folder_name):
    """
    將 vectors_data 轉換為 DataFrame 並保存為 CSV 檔案。

    :param vectors_data: 要保存的向量資料
    :param input_path: 輸入檔案的路徑，用於確定 CSV 檔案的保存位置
    :param folder_name: 資料夾名稱，用於命名 CSV 檔案
    """
    import pandas as pd
    # 將 vectors_data 轉換為 DataFrame
    df = pd.DataFrame(vectors_data)

    # 將 DataFrame 保存為 CSV 檔案
    csv_output_path = os.path.join(os.path.dirname(input_path), f"{folder_name}_vectors_data.csv")
    df.to_csv(csv_output_path, index=False, float_format='%.8f')  # 使用 float_format 格式化小數點表示法
    print(Fore.GREEN + f"CSV file saved to: {csv_output_path}")

def process_video(input_path, output_path, csv_file, folder_name):
    # 初始化 frame_number
    frame_number = 0

    # 初始化總向量動能列表
    total_vector_energy_list = []
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
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*SCALE, height*SCALE))
    
    

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

        # 初始化總向量動能
        total_vector_energy = 0.0

        # Calculate vectors only for landmarks that appeared in the last frame
        for key, value in current_frame_landmarks.items():
            if key in last_frame_landmarks:
                prev_value = last_frame_landmarks[key]
                vector = np.array(value) - np.array(prev_value)
                
                if np.all(np.abs(vector) <= THRESHOLD):  # Check if all components of the vector are within the threshold
                    vectors_list.append((key, vector))
                    start_point = (int(prev_value[0] * width), int(prev_value[1] * height))
                    end_point = (int(value[0] * width), int(value[1] * height))
                    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)  # Red line for vector
                    
                    # 計算向量動能並累加
                    vector_energy = np.sum(vector ** 2)
                    total_vector_energy += vector_energy

                    # 打印向量資訊
                    
                    #print(Fore.GREEN + f"Frame: {frame_number}, Landmark: {key}, Vector: {vector}, Total Vector Energy: {vector_energy * 1000:.2f}")                    # 保存向量資訊到資料結構
                    vectors_data.append({
                        'frame': frame_number,
                        'landmark': key,
                        'vector': vector.tolist(),
                        'total_vector': vector_energy
                    })
  
        # 保存總向量動能到列表
        total_vector_energy_list.append(total_vector_energy)

        # 在畫面上顯示總向量動能
        cv2.putText(frame, f"{total_vector_energy:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),1, cv2.LINE_AA)

        # 更新 last_frame_landmarks
        last_frame_landmarks = current_frame_landmarks

        # 更新 frame_number
        frame_number += 1


        # Resize frame to double the size
        frame_resized = cv2.resize(frame, (width *SCALE, height *SCALE))

        last_frame_landmarks = current_frame_landmarks
        out.write(frame_resized)
        cv2.imshow('Video Feed', frame_resized)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 繪製總向量動能隨時間變化的曲線圖 並添加folder_name並使用,folder_name作為檔名存檔
    plt.figure()
    plt.plot(total_vector_energy_list)
    plt.xlabel('Frame Number : '+folder_name)
    plt.ylabel('Total Vector Energy')
    plt.title('Total Vector Energy Over Time')
    #plt.savefig(folder_name+'.png')

    #plt.show()
    # 將 vectors_data 轉換為 DataFrame
    save_vectors_to_csv(vectors_data, input_path, folder_name)

    # 刪除原始 MP4 檔案
    if os.path.exists(input_path):
        os.remove(input_path)
        print(Fore.RED + f"Original video file {input_path} has been deleted.")

    return vectors_list

def process_all_videos(base_dir, annotations_dir):
    # 統計需要處理的資料夾數量
    total_dirs = 0
    for subdir in ['test', 'dev', 'train']:
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.exists(subdir_path):
            for _, dirs, _ in os.walk(subdir_path):
                total_dirs += len(dirs)
    
    print(Fore.YELLOW + f"Total directories to process: {total_dirs}")
    
    processed_dirs = 0
    
    for subdir in ['test', 'dev', 'train']:
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(subdir_path):
            print(Fore.RED + f"Directory {subdir_path} does not exist.")
            continue
        
        # 根據 subdir 載入對應的 CSV 檔案
        if subdir == 'test':
            csv_file = os.path.join(annotations_dir, 'PHOENIX-2014-T.test.corpus.csv')
        elif subdir == 'dev':
            csv_file = os.path.join(annotations_dir, 'PHOENIX-2014-T.dev.corpus.csv')
        elif subdir == 'train':
            csv_file = os.path.join(annotations_dir, 'PHOENIX-2014-T.train-complex-annotation.corpus.csv')
        
        for root, dirs, files in os.walk(subdir_path):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                video_file = None
                
                # 檢查是否有 .mp4 檔案
                for file in os.listdir(dir_path):
                    if file.endswith('.mp4'):
                        video_file = os.path.join(dir_path, file)
                        break
                
                # 如果沒有 .mp4 檔案，檢查是否有 .png 檔案並轉換成 .mp4
                if not video_file:
                    output_mp4 = os.path.join(dir_path, f"{dir_name}_output.mp4")
                    if pngs_to_mp4(dir_path, output_mp4):
                        video_file = output_mp4
                        print(Fore.YELLOW + f"Converted PNGs to video: {video_file}")
                    else:
                        print(Fore.RED + f"No PNG files found in {dir_path}")
                        continue
                
                # 處理影片
                output_video = os.path.splitext(video_file)[0] + "_output.mp4"
                folder_name = os.path.basename(dir_path)
                print(Fore.YELLOW + f"Processing video: {video_file}")
                vectors = process_video(video_file, output_video, csv_file, folder_name)
                print(Fore.GREEN + f"Total vectors captured: {len(vectors)}")
                
                # 更新並打印進度
                processed_dirs += 1
                progress = (processed_dirs / total_dirs) * 100
                print(Fore.CYAN + f"Progress: {processed_dirs}/{total_dirs} ({progress:.2f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', help='Base directory containing dev, test, and train folders')
    parser.add_argument('--csv', dest='csv_file', help='Path to the CSV file')
    args = parser.parse_args()

    base_dir = args.base_dir
    csv_file = args.csv_file

    print(Fore.YELLOW + "Parsed Arguments:")
    print(Fore.YELLOW + f"  Base Directory: {base_dir}")
    print(Fore.YELLOW + f"  CSV File: {csv_file}")

    print("\n=====================================\n")
    process_all_videos(base_dir, csv_file)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--in', dest='input', help='Input video path or PNG folder')
#     parser.add_argument('--out', dest='output', help='Output video path', default=None)
#     parser.add_argument('--csv', dest='csv_file', help='Path to the CSV file')
#     args = parser.parse_args()
#     input_path = args.input
#     output_video = args.output or os.path.splitext(input_path)[0] + "_output.mp4"
#     # 更清楚地打印每個參數
#     print(Fore.YELLOW + "Parsed Arguments:")
#     print(Fore.YELLOW + f"  Input Path: {args.input}")
#     print(Fore.YELLOW + f"  Output Path: {args.output}")
#     print(Fore.YELLOW + f"  CSV File: {args.csv_file}")
#     folder_name=input_path.split('/')[-1]
#     print(Fore.YELLOW + f"  Folder Name: {folder_name}")


#     if os.path.isdir(input_path):
#         temp_mp4 = "temp_video.mp4"
#         pngs_to_mp4(input_path, temp_mp4)
#         vectors = process_video(temp_mp4, output_video, args.csv_file,folder_name)
#         os.remove(temp_mp4)
#     else:
#         vectors = process_video(input_path, output_video, args.csv_file, folder_name)
    
#     print(Fore.GREEN + f"Total vectors captured: {len(vectors)}")