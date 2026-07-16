document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const resultsSection = document.getElementById('results-section');
    const resultsTable = document.getElementById('results-table').querySelector('tbody');
    const batchStats = document.getElementById('batch-stats');
    
    // Drag and drop events
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    btnBrowse.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });
    
    async function handleFile(file) {
        if (!file.name.match(/\.(txt|csv|jsonl)$/i)) {
            alert("Unsupported file format. Please use .txt, .csv, or .jsonl");
            return;
        }
        
        uploadZone.innerHTML = '<div class="spinner mx-auto mb-3" style="margin:0 auto 1rem;"></div><h3>Analyzing File...</h3><p class="text-muted">This may take a moment.</p>';
        
        const reader = new FileReader();
        reader.onload = async (e) => {
            const content = e.target.result;
            const texts = parseFile(content, file.name);
            
            if (texts.length === 0) {
                alert("No valid texts found in file.");
                resetUploadZone();
                return;
            }
            
            // Chunking for API safety (max 50 at a time)
            const MAX_BATCH = 50;
            const chunks = [];
            for (let i = 0; i < texts.length; i += MAX_BATCH) {
                chunks.push(texts.slice(i, i + MAX_BATCH));
            }
            
            try {
                let allResults = [];
                let summary = { total: 0, human: 0, ai: 0, uncertain: 0 };
                
                for (let i=0; i < chunks.length; i++) {
                    const data = await fetchApi('/analyze/batch', {
                        method: 'POST',
                        body: JSON.stringify({ texts: chunks[i] })
                    });
                    
                    allResults = allResults.concat(data.results);
                    summary.total += data.summary.total;
                    summary.human += data.summary.human;
                    summary.ai += data.summary.ai;
                    summary.uncertain += data.summary.uncertain;
                }
                
                displayResults(allResults, summary);
                
            } catch (error) {
                alert("Error analyzing batch: " + error.message);
                resetUploadZone();
            }
        };
        reader.readAsText(file);
    }
    
    function parseFile(content, filename) {
        const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        
        if (filename.endsWith('.jsonl')) {
            return lines.map(l => {
                try { return JSON.parse(l).text; } catch(e) { return null; }
            }).filter(t => t && t.length > 10);
        }
        
        if (filename.endsWith('.csv')) {
            // Very naive CSV parsing, assuming first column or a 'text' column
            // In a real app, use a CSV parser library
            const headers = lines[0].toLowerCase().split(',');
            let textIdx = headers.indexOf('text');
            if (textIdx === -1) textIdx = 0;
            
            return lines.slice(1).map(l => {
                const cols = l.split(',');
                return cols[textIdx] ? cols[textIdx].replace(/^["']|["']$/g, '') : null;
            }).filter(t => t && t.length > 10);
        }
        
        // txt
        return lines.filter(l => l.length > 10);
    }
    
    function resetUploadZone() {
        uploadZone.innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <h3>Drag & Drop files here</h3>
            <p class="text-muted mb-3">or click to browse</p>
            <button class="btn btn-primary" id="btn-browse-retry">Select File</button>
        `;
        document.getElementById('btn-browse-retry').addEventListener('click', () => fileInput.click());
    }
    
    let currentResults = [];
    
    function displayResults(results, summary) {
        currentResults = results;
        uploadZone.style.display = 'none';
        resultsSection.classList.remove('hidden');
        
        // Stats
        batchStats.innerHTML = `
            <div class="card stat-card">
                <div class="stat-value text-gradient">${summary.total}</div>
                <div>Total Analyzed</div>
            </div>
            <div class="card stat-card" style="border-color: rgba(16, 185, 129, 0.3);">
                <div class="stat-value" style="color: var(--color-success);">${summary.human}</div>
                <div>Likely Human</div>
            </div>
            <div class="card stat-card" style="border-color: rgba(239, 68, 68, 0.3);">
                <div class="stat-value" style="color: var(--color-danger);">${summary.ai}</div>
                <div>Likely AI</div>
            </div>
            <div class="card stat-card" style="border-color: rgba(245, 158, 11, 0.3);">
                <div class="stat-value" style="color: var(--color-warning);">${summary.uncertain}</div>
                <div>Uncertain</div>
            </div>
        `;
        
        // Table
        resultsTable.innerHTML = '';
        results.forEach(r => {
            const tr = document.createElement('tr');
            
            let vClass = 'badge-neutral';
            const vLower = r.verdict.toLowerCase();
            if (vLower.includes('human')) vClass = 'badge-success';
            else if (vLower.includes('ai')) vClass = 'badge-danger';
            else vClass = 'badge-warning';
            
            tr.innerHTML = `
                <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${r.text}">${r.text}</td>
                <td><span class="badge ${vClass}">${r.verdict}</span></td>
                <td>${r.confidence}</td>
                <td>${r.authenticity_score}%</td>
                <td>${(r.synthetic_probability * 100).toFixed(1)}%</td>
            `;
            resultsTable.appendChild(tr);
        });
    }
    
    // Export CSV
    document.getElementById('btn-export').addEventListener('click', () => {
        if (!currentResults.length) return;
        
        const headers = ["Text", "Verdict", "Confidence", "Authenticity Score", "Synthetic Probability"];
        const csvRows = [headers.join(',')];
        
        currentResults.forEach(r => {
            const cleanText = r.text.replace(/"/g, '""');
            csvRows.push(`"${cleanText}","${r.verdict}","${r.confidence}",${r.authenticity_score},${r.synthetic_probability}`);
        });
        
        const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.setAttribute('hidden', '');
        a.setAttribute('href', url);
        a.setAttribute('download', 'prism_batch_results.csv');
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
});
