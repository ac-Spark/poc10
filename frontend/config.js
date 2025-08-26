// 從環境變量或默認值讀取配置
const CONFIG = {
    API_BASE_URL: 'http://localhost:8860',
    FRONTEND_PORT: 8861
};

// 如果運行在開發環境，可以從 .env 讀取配置
if (typeof process !== 'undefined' && process.env) {
    CONFIG.API_BASE_URL = process.env.API_BASE_URL || CONFIG.API_BASE_URL;
    CONFIG.FRONTEND_PORT = process.env.FRONTEND_PORT || CONFIG.FRONTEND_PORT;
}

export default CONFIG;