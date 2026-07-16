document.addEventListener('DOMContentLoaded', async () => {
    try {
        const data = await fetchApi('/metrics');
        
        // Populate stats
        document.getElementById('stat-accuracy').textContent = `${(data.accuracy * 100).toFixed(1)}%`;
        document.getElementById('stat-f1').textContent = data.f1.toFixed(3);
        document.getElementById('stat-roc').textContent = data.roc_auc.toFixed(3);
        document.getElementById('stat-fpr').textContent = `${(data.fpr * 100).toFixed(1)}%`;
        
        // Confusion Matrix
        if (data.confusion_matrix) {
            document.getElementById('cm-tn').textContent = data.confusion_matrix.tn;
            document.getElementById('cm-fp').textContent = data.confusion_matrix.fp;
            document.getElementById('cm-fn').textContent = data.confusion_matrix.fn;
            document.getElementById('cm-tp').textContent = data.confusion_matrix.tp;
        }
        
        // Info
        if (data.model_info) {
            document.getElementById('info-version').textContent = data.model_info.version;
            document.getElementById('info-mode').textContent = data.model_info.mode;
            document.getElementById('info-classifier').textContent = data.model_info.classifier;
            document.getElementById('info-features').textContent = data.model_info.features_count;
            document.getElementById('info-rows').textContent = data.model_info.train_rows.toLocaleString();
        }
        
    } catch (error) {
        console.error("Failed to load metrics:", error);
    }
});
