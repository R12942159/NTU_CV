#!/bin/bash

MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

# Google Drive file IDs
FILE1_ID="15Vvl83uV6jWcspzLRkJVdExfPRlRjF2z"
FILE2_ID="1XYeU-v2RU028Hm1SbcUrIIxsabWw20LK"
FILE3_ID="16bfhMTLaHzfzetUPWrYqq7VJfQykBGUd"

download_file () {
    FILE_ID=$1
    FILE_NAME=$2

    echo "Downloading $FILE_NAME..."
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
    CONFIRM=$(awk '/download/ {print $NF}' ./cookie)
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o "${MODEL_DIR}/${FILE_NAME}"
    echo "$FILE_NAME downloaded."
}

download_file $FILE1_ID "resnet18-1543-0.047488-maskIoU-0.934494.pth"
download_file $FILE2_ID "nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth"
download_file $FILE3_ID "convnext_tiny-znorm-best.pth"

rm -f ./cookie

echo "All files downloaded!"