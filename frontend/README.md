# Hunyuan3D Frontend

前端獨立目錄，提供 Web 界面用於 3D 模型生成。

## 文件結構

```
frontend/
├── index.html          # 主要的 HTML 前端頁面
├── server.py           # 簡單的 HTTP 服務器
├── config.js           # 配置文件
├── .env                # 環境變量配置
└── README.md           # 說明文檔
```

## 運行方式

### 1. Docker 運行 (推薦)
```bash
# 從根目錄運行
./run-docker.sh
# 或
docker-compose up -d
```

### 2. 本地運行
```bash
cd frontend
pip install python-dotenv
python server.py
```

## 配置

前端配置通過 `frontend/.env` 文件進行：

```
FRONTEND_PORT=8861
API_BASE_URL=http://localhost:8860
DEV_MODE=true
DEBUG=false
```

## 功能特性

- 📁 多視圖圖片上傳 (前視圖、後視圖、左視圖、右視圖)
- 🖼️ 圖片預覽和拖拽上傳
- 📊 實時進度顯示與狀態更新
- 🎯 3D 模型預覽 (GLB 格式)
- 🎨 材質生成選項
- 📦 多格式輸出支持 (GLB / OBJ)
- 💾 自動文件下載
- 🔄 重新生成功能
- 🌙 深色主題界面

## API 依賴

前端需要後端 API 服務運行在配置的端口上（默認 8860）。

## 支持的輸出格式

- **GLB (GLTF Binary)**: 適合 Web 展示，支持瀏覽器預覽，包含完整材質信息
- **OBJ (Wavefront)**: 通用 3D 格式，兼容大多數 3D 軟件，需要下載後在專業軟件中查看

## 使用說明

1. 上傳至少一張圖片（支持前、後、左、右四個視圖）
2. 選擇是否生成材質（會增加處理時間）
3. 選擇輸出格式（GLB 或 OBJ）
4. 點擊生成按鈕
5. 等待處理完成後下載或預覽結果