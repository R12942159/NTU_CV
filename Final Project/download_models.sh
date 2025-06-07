#!/bin/bash

MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

# Google Drive file IDs
FILE1_ID="15Vvl83uV6jWcspzLRkJVdExfPRlRjF2z"
FILE2_ID="1XYeU-v2RU028Hm1SbcUrIIxsabWw20LK"
FILE3_ID="16bfhMTLaHzfzetUPWrYqq7VJfQykBGUd"

download_file() {
    FILE_ID=$1
    FILE_NAME=$2
    
    echo "Downloading $FILE_NAME..."
    
    # First attempt - direct download
    wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
         "https://drive.google.com/uc?export=download&id=${FILE_ID}" -O- | \
         sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p' > /tmp/confirm.txt
    
    # Get confirmation code
    CONFIRM=$(cat /tmp/confirm.txt)
    
    # Download with confirmation
    if [ -n "$CONFIRM" ]; then
        wget --load-cookies /tmp/cookies.txt \
             "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
             -O "${MODEL_DIR}/${FILE_NAME}"
    else
        # Try direct download for smaller files
        wget "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
             -O "${MODEL_DIR}/${FILE_NAME}"
    fi
    
    # Clean up
    rm -f /tmp/cookies.txt /tmp/confirm.txt
    
    # Check if we got an HTML file instead of the model
    if file "${MODEL_DIR}/${FILE_NAME}" | grep -q HTML; then
        echo "Error: Downloaded HTML instead of model file for $FILE_NAME"
        echo "Please download manually from: https://drive.google.com/file/d/${FILE_ID}/view"
        rm "${MODEL_DIR}/${FILE_NAME}"
        return 1
    fi
    
    echo "$FILE_NAME downloaded successfully."
}

# Alternative method using gdown (more reliable)
use_gdown() {
    echo "Checking for gdown..."
    if ! command -v gdown &> /dev/null; then
        echo "gdown not found. Installing..."
        pip install gdown
    fi
    
    echo "Using gdown to download models..."
    gdown --id $FILE1_ID -O "${MODEL_DIR}/resnet18-1543-0.047488-maskIoU-0.934494.pth"
    gdown --id $FILE2_ID -O "${MODEL_DIR}/nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth"
    gdown --id $FILE3_ID -O "${MODEL_DIR}/convnext_tiny-znorm-best.pth"
}

# Check if wget is available
if ! command -v wget &> /dev/null; then
    echo "wget not found. Using gdown method instead..."
    use_gdown
else
    # Try wget method first
    download_file $FILE1_ID "resnet18-1543-0.047488-maskIoU-0.934494.pth"
    download_file $FILE2_ID "nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth"
    download_file $FILE3_ID "convnext_tiny-znorm-best.pth"
    
    # Check if downloads were successful
    if [ ! -f "${MODEL_DIR}/resnet18-1543-0.047488-maskIoU-0.934494.pth" ] || \
       [ $(stat -f%z "${MODEL_DIR}/resnet18-1543-0.047488-maskIoU-0.934494.pth" 2>/dev/null || stat -c%s "${MODEL_DIR}/resnet18-1543-0.047488-maskIoU-0.934494.pth" 2>/dev/null) -lt 1000000 ]; then
        echo "Download failed. Trying gdown method..."
        use_gdown
    fi
fi

# Verify downloads
echo ""
echo "Verifying downloads..."
ls -lh "${MODEL_DIR}/"

echo ""
echo "Download complete!"