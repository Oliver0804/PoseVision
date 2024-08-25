![https://github.com/Oliver0804/PoseVision/blob/main/output.gif](https://github.com/Oliver0804/PoseVision/blob/main/output.gif)

# 安裝與測試指南
本指南將引導您在conda環境中安裝並測試使用Python 3.11的程式。請按照以下步驟進行操作。

## 步驟 1：安裝 conda
前往 Anaconda 官方網站 下載適用於您作業系統的最新版本的 Anaconda。
遵循官方網站上的安裝指示進行安裝。
## 步驟 2：建立 conda 環境
開啟終端機（命令提示字元）。
執行以下命令以建立名為 myenv 的 conda 環境：
```bash
conda create -n myenv python=3.11
```
啟用 conda 環境：
```bash
conda activate myenv
```
## 步驟 3：安裝所需套件
在 conda 環境中，安裝所需的套件以便執行程式。

```
pip install -r  
```
## 步驟 4：執行程式
在終端機中，切換到包含程式檔案的目錄。
執行以下命令以運行程式：
```bash
python main.py --in <input_video_path> --out <output_video_path>
 python main.py --base_dir ../fullFrame-210x260px/ --csv ./annotations/manual
```
請將 <input_video_path> 替換為您要處理的影片路徑，並將 <output_video_path> 替換為您想要的輸出影片名稱。


您將看到程式正在運行並處理影片，同時在終端機上顯示關鍵點的移動向量數值。

按下 q 鍵停止程式運行。


資料夾最後不需要帶"/"
```bash
python main.py --in ./28October_2009_Wednesday_tagesschau-4534
```


## pt
會針對影像向量by frame記入成pt檔案

## 備註
如果您想測試其他影片，請確保您的影片具有支援的格式和編解碼器。如果遇到播放或編碼問題，請檢查您的系統是否具有所需的多媒體套件和編解碼器。
請確保您的系統上已經安裝了適用於 Python 3.11 的 conda 環境，並且按照指南中的步驟進行操作。
這樣，您就可以在conda環境中安裝並測試使用Python 3.11的程式。如果您遇到任何問題，請隨時聯繫我們，我們很樂意為您提供協助。
