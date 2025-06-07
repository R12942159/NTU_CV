### 環境設置

    請先使用 virtual environment 下載 requirements.txt 中所列出的套件。
    （給各位參考一下，我在本次 project 中使用的 python 版本為 3.13.3）

### 模型參數下載

   專案根目錄包含一個下載腳本 `download_models.sh`，執行該腳本會自動從 Google Drive 下載三個模型檔案到 `Final Project/models` 資料夾。

   確保你的環境有安裝 `curl`，在終端機（Terminal）執行：

   ```bash
   bash download_models.sh

### 程式架構說明

1. src/

    src/ 內存放所有和本次 project 相關的程式碼。run.py 為主程式。
    而 src/modules/ 內存放所有和虹膜辨識功能相關的實做，主要的 code 都在 irisRecognition 這個 class 裡面（可以和 run.py 交替著看）

2. models/

    存放各個模型所需的 pre-trained weight。

3. filters_pt/

    存放 pre-trained filters。

4. input_list/

    存放測資。

5. test/

    存放辨識結果。(裡面 result_*.txt 的檔案是上傳到 Codabench 的結果)

6. cfg.yaml

    此檔案中紀錄程式執行過程中所需要的參數設定。

### 執行指令

python3 ./src/run.py --input input_list/list_CASIA-Iris-Thousand.txt --output test/list_CASIA-Iris-Thousand.txt
python3 ./src/run.py --input input_list/list_CASIA-Iris-Lamp.txt --output test/list_CASIA-Iris-Lamp.txt
python3 ./src/run.py --input input_list/list_Ganzin-J7EF-Gaze.txt --output test/list_Ganzin-J7EF-Gaze.txt

### 使用 eval.py 計算 d'score

python3 ./src/eval.py --input test/list_CASIA-Iris-Thousand.txt
python3 ./src/eval.py --input test/list_CASIA-Iris-Lamp.txt
python3 ./src/eval.py --input test/list_Ganzin-J7EF-Gaze.txt

### Reference
    如果有對任何一個地方的環節不太清楚可以參考下面 github 連結。

    OpenSourceIrisRecognition 首頁://github.com/CVRL/OpenSourceIrisRecognition
    我參照的程式碼（在架構上我幾乎沒有做更動）：
        1. https://github.com/CVRL/OpenSourceIrisRecognition/tree/main/methods/TripletNN/Python
        2. https://github.com/CVRL/OpenSourceIrisRecognition/tree/main/methods/HDBIF/Python
