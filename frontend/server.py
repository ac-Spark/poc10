#!/usr/bin/env python3
"""
簡單的HTTP服務器，用於運行前端頁面
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from dotenv import load_dotenv

# 加載環境變量
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def run_server():
    # 切換到前端目錄
    frontend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(frontend_dir)
    
    # 從環境變量獲取端口
    port = int(os.getenv('FRONTEND_PORT', 8861))
    
    server = HTTPServer(('0.0.0.0', port), CORSHTTPRequestHandler)
    
    print(f"🌐 前端服務器啟動在 http://localhost:{port}")
    print(f"📁 服務目錄: {frontend_dir}")
    print("按 Ctrl+C 停止服務器")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服務器已停止")
        server.server_close()

if __name__ == "__main__":
    run_server()