const API_BASE = 'http://localhost:8000/api';

/**
 * Shared fetch wrapper with basic error handling
 */
async function fetchApi(endpoint, options = {}) {
    try {
        const url = `${API_BASE}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };
        const res = await fetch(url, { ...defaultOptions, ...options });
        
        if (!res.ok) {
            let errorMsg = `HTTP Error ${res.status}`;
            try {
                const data = await res.json();
                if (data.detail) errorMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
            } catch (e) {}
            throw new Error(errorMsg);
        }
        
        return await res.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Check API health on load
 */
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await fetchApi('/health');
    } catch (error) {
        console.warn('Backend may be offline:', error.message);
    }
});
