#!/bin/bash

echo "用 Poetry 安裝一般套件..."

# 先用 poetry add 安裝一般套件（跳過特殊指令）
while IFS= read -r line; do
    # 跳過空行、註解、pip 指令
    if [[ -z "$line" || "$line" == \#* || "$line" == --* || "$line" == -* ]]; then
        continue
    fi
    
    package=$(echo "$line" | sed 's/[[:space:]]*$//')
    echo "poetry add $package"
    poetry add "$package"
    
done < requirements.txt

echo "用 pip 補充安裝（處理特殊來源）..."
poetry run pip install -r requirements.txt

echo "完成！"