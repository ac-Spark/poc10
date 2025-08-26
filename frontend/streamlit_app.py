import streamlit as st
import requests
import time
import base64
from io import BytesIO

# 配置頁面
st.set_page_config(
    page_title="Hunyuan3D 2.1", 
    page_icon="🎨", 
    layout="wide"
)

# API 設定 - 支持 Docker 環境
import os
API_BASE = os.getenv('API_BASE_URL', 'http://localhost:8860')

# 標題
st.title("🎨 Hunyuan3D 2.1")
st.subheader("將圖片轉換成 3D 模型")

# 側邊欄參數
with st.sidebar:
    st.header("⚙️ 生成參數")
    num_steps = st.slider("推理步數", 1, 20, 5)
    texture = st.checkbox("生成材質", value=False)
    seed = st.number_input("隨機種子", value=1234, min_value=0)

# 主要區域
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 上傳圖片")
    
    uploaded_file = st.file_uploader(
        "選擇圖片文件", 
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="支援 PNG, JPG, JPEG, WebP 格式"
    )
    
    if uploaded_file is not None:
        # 顯示預覽
        st.image(uploaded_file, caption="上傳的圖片", use_container_width=True)
        
        # 生成按鈕
        if st.button("🚀 生成 3D 模型", type="primary", use_container_width=True):
            with st.spinner("正在處理..."):
                try:
                    # 準備文件數據
                    files = {"image_front": uploaded_file}
                    data = {
                        "num_inference_steps": num_steps,
                        "texture": texture,
                        "seed": seed
                    }
                    
                    # 提交任務
                    response = requests.post(f"{API_BASE}/send_file", files=files, data=data)
                    
                    if response.status_code == 200:
                        task_id = response.json()["uid"]
                        st.success(f"任務已提交！ID: {task_id}")
                        
                        # 進度條
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 輪詢狀態
                        while True:
                            status_response = requests.get(f"{API_BASE}/status/{task_id}")
                            status_data = status_response.json()
                            
                            if status_data["status"] == "processing":
                                progress_bar.progress(30)
                                status_text.text("🔄 正在生成 3D 形狀...")
                            elif status_data["status"] == "texturing":
                                progress_bar.progress(70)
                                status_text.text("🎨 正在添加材質...")
                            elif status_data["status"] == "completed":
                                progress_bar.progress(100)
                                status_text.text("✅ 生成完成！")
                                
                                # 保存結果到 session state
                                st.session_state.model_data = status_data["model_base64"]
                                st.session_state.task_id = task_id
                                st.rerun()
                                break
                            elif status_data["status"] == "error":
                                st.error(f"❌ 生成失敗: {status_data.get('message', '未知錯誤')}")
                                break
                            
                            time.sleep(2)
                    
                    else:
                        st.error("❌ 提交失敗")
                        
                except Exception as e:
                    st.error(f"❌ 錯誤: {str(e)}")

with col2:
    st.header("🎯 生成結果")
    
    # 顯示結果
    if hasattr(st.session_state, 'model_data') and st.session_state.model_data:
        st.success("🎉 3D 模型生成完成！")
        
        # 下載按鈕
        model_bytes = base64.b64decode(st.session_state.model_data)
        st.download_button(
            label="💾 下載 GLB 文件",
            data=model_bytes,
            file_name=f"model_{st.session_state.task_id}.glb",
            mime="application/octet-stream",
            use_container_width=True
        )
        
        # 3D 預覽 (HTML)
        st.subheader("👁️ 3D 預覽")
        
        # 創建臨時文件 URL
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.glb') as tmp_file:
            tmp_file.write(model_bytes)
            tmp_path = tmp_file.name
        
        # 生成 model-viewer HTML
        model_viewer_html = f"""
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        <div style="display: flex; justify-content: center;">
            <model-viewer 
                src="data:application/octet-stream;base64,{st.session_state.model_data}"
                alt="Generated 3D Model"
                auto-rotate 
                camera-controls
                style="width: 100%; height: 400px; background-color: #f0f0f0;">
            </model-viewer>
        </div>
        """
        
        st.components.v1.html(model_viewer_html, height=450)
        
        # 清除按鈕
        if st.button("🔄 生成新模型", use_container_width=True):
            if hasattr(st.session_state, 'model_data'):
                del st.session_state.model_data
            if hasattr(st.session_state, 'task_id'):
                del st.session_state.task_id
            st.rerun()
    
    else:
        st.info("👆 請先上傳圖片並點擊生成")

# 頁腳
st.markdown("---")
st.markdown("🔗 **API 端點**: http://localhost:8860 | 📚 **文檔**: http://localhost:8860/docs")