#!/bin/bash

# 檔案 1：input_list.zip
echo "Downloading input_list.zip..."
gdown --id 1homhvqvqbq-31iazp3IY9JNoXGdqVne2 -O input_list.zip

# 檔案 2：dataset.zip
echo "Downloading dataset.zip..."
gdown --id 1VkzwCkA5g6do93VjTfNKwL_yWkq0-QGw -O dataset.zip

# 解壓縮
echo "Unzipping input_list.zip..."
unzip -o input_list.zip

echo "Unzipping dataset.zip..."
unzip -o dataset.zip

# 刪除壓縮檔
echo "Cleaning up zip files..."
rm input_list.zip dataset.zip

echo "✅ All done."
