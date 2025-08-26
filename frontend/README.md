# Hunyuan3D Frontend

前端獨立目錄，提供 Web 界面用於 3D 模型生成。

## 文件結構

```
frontend/
├── index.html          # 主要的 HTML 前端頁面
├── streamlit_app.py    # Streamlit 版本的前端
├── server.py           # 簡單的 HTTP 服務器
├── config.js           # 配置文件
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

### 3. Streamlit 版本
```bash
cd frontend
pip install streamlit python-dotenv
streamlit run streamlit_app.py --server.port 8861
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

- 📁 拖拽上傳圖片
- 🖼️ 圖片預覽
- 📊 實時進度顯示
- 🎯 3D 模型預覽
- 💾 GLB 文件下載
- 🔄 重新生成功能

## API 依賴

前端需要後端 API 服務運行在配置的端口上（默認 8860）。