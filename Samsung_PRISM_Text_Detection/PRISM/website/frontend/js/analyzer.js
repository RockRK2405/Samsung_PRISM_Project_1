document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('text-input');
    const fileInput = document.getElementById('file-input');
    const fileDropZone = document.getElementById('file-drop-zone');
    const mediaPreview = document.getElementById('media-preview');
    const charCount = document.getElementById('char-count');
    const wordCount = document.getElementById('word-count');
    const btnAnalyze = document.getElementById('btn-analyze');
    const analyzeSpinner = document.getElementById('analyze-spinner');
    const errorMsg = document.getElementById('error-msg');
    const resultsPanel = document.getElementById('results-panel');
    
    // Sample texts
    const humanSample = "The concept of burstiness in writing refers to the variation in sentence length and structure. Human writers naturally vary their sentences. They might use a short, punchy statement. Then, they follow it up with a much longer, more complex sentence that elaborates on the initial point, weaving together different clauses and ideas to provide depth and context to the reader. This natural ebb and flow keeps the reader engaged and makes the text feel alive.";
    
    const aiSample = "Artificial intelligence has become increasingly prevalent in modern society. Furthermore, it plays a vital role in streamlining various industrial processes. In addition to improving efficiency, it also provides significant cost savings for businesses. However, it is important to recognize that these advancements come with certain challenges. In conclusion, while AI offers numerous benefits, careful consideration must be given to its implementation.";
    
    document.getElementById('btn-sample-human').addEventListener('click', () => {
        textInput.value = humanSample;
        updateCounts();
    });
    
    document.getElementById('btn-sample-ai').addEventListener('click', () => {
        textInput.value = aiSample;
        updateCounts();
    });

    // Update counts
    function updateCounts() {
        const text = textInput.value;
        charCount.textContent = `${text.length} characters`;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        wordCount.textContent = `${words} words`;
    }
    
    textInput.addEventListener('input', updateCounts);
    
    // File Drag and Drop logic
    if (fileDropZone) {
        fileDropZone.addEventListener('click', () => fileInput.click());
        fileDropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileDropZone.style.borderColor = 'var(--color-brand)';
        });
        fileDropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            fileDropZone.style.borderColor = 'var(--border-strong)';
        });
        fileDropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            fileDropZone.style.borderColor = 'var(--border-strong)';
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelection(fileInput.files[0]);
            }
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFileSelection(fileInput.files[0]);
            }
        });
    }

    function handleFileSelection(file) {
        fileDropZone.querySelector('p').innerHTML = `File selected: <strong>${file.name}</strong>`;
        mediaPreview.innerHTML = '';
        mediaPreview.classList.remove('hidden');

        const fileType = file.type;
        const fileUrl = URL.createObjectURL(file);

        if (fileType.startsWith('image/')) {
            mediaPreview.innerHTML = `<img src="${fileUrl}" style="max-width: 100%; max-height: 300px; border-radius: var(--border-radius);" alt="Preview"/>`;
        } else if (fileType.startsWith('audio/')) {
            mediaPreview.innerHTML = `<audio controls style="width: 100%;"><source src="${fileUrl}" type="${fileType}"></audio>`;
        } else if (fileType.startsWith('video/')) {
            mediaPreview.innerHTML = `<video controls style="max-width: 100%; max-height: 300px; border-radius: var(--border-radius);"><source src="${fileUrl}" type="${fileType}"></video>`;
        } else {
            mediaPreview.innerHTML = `<div style="padding: 2rem;"><i class="fas fa-file-alt" style="font-size: 3rem; color: var(--text-muted); margin-bottom: 1rem;"></i><p>Document ready for analysis</p></div>`;
        }
    }

    // Analyze
    btnAnalyze.addEventListener('click', async () => {
        const text = textInput.value.trim();
        const hasFile = fileInput && fileInput.files.length > 0;
        
        if (!hasFile && text.length < 50) {
            errorMsg.textContent = "Please enter at least 50 characters, or upload a file.";
            errorMsg.style.display = 'block';
            return;
        }
        
        errorMsg.style.display = 'none';
        btnAnalyze.disabled = true;
        analyzeSpinner.classList.remove('hidden');
        resultsPanel.style.opacity = '0.5';
        resultsPanel.style.pointerEvents = 'none';
        
        try {
            let data;
            if (hasFile) {
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(errText || 'Upload failed');
                }
                data = await response.json();
                
                // Update text input with extracted text
                if (data.extracted_text) {
                    textInput.value = data.extracted_text;
                    updateCounts();
                }
            } else {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(errText || 'Analysis failed');
                }
                data = await response.json();
            }
            
            displayResults(data);
            
            resultsPanel.style.opacity = '1';
            resultsPanel.style.pointerEvents = 'auto';
        } catch (error) {
            errorMsg.textContent = error.message;
            errorMsg.style.display = 'block';
        } finally {
            btnAnalyze.disabled = false;
            analyzeSpinner.classList.add('hidden');
        }
    });
    
    let radarChart = null;
    
    function displayResults(data) {
        // Verdict Badge
        const badge = document.getElementById('verdict-badge');
        badge.textContent = data.verdict;
        badge.className = 'badge text-lg mb-2';
        
        const vLower = data.verdict.toLowerCase();
        if (vLower.includes('human')) badge.classList.add('badge-success');
        else if (vLower.includes('ai')) badge.classList.add('badge-danger');
        else badge.classList.add('badge-warning');
        
        document.getElementById('confidence-text').textContent = `${data.confidence} confidence`;
        
        // Gauge
        document.getElementById('authenticity-score').textContent = `${data.authenticity_score}%`;
        const arc = document.getElementById('authenticity-arc');
        arc.style.strokeDasharray = `${data.authenticity_score}, 100`;
        
        if (data.authenticity_score > 65) arc.style.stroke = 'var(--color-success)';
        else if (data.authenticity_score < 35) arc.style.stroke = 'var(--color-danger)';
        else arc.style.stroke = 'var(--color-warning)';
        
        // Explanation
        document.getElementById('explanation-text').textContent = data.explanation;
        
        // Signals
        const sigs = data.signals || {};
        
        // Sig 1 (Stylometric)
        document.getElementById('sig1-label').textContent = sigs.stylometric?.label || '--';
        const s1Fill = document.getElementById('sig1-fill');
        s1Fill.style.width = `${(sigs.stylometric?.score || 0) * 100}%`;
        s1Fill.className = 'progress-fill ' + getFillClass(sigs.stylometric?.score);
        
        // Sig 2 (Perplexity)
        document.getElementById('sig2-label').textContent = sigs.perplexity?.label || '--';
        const s2Fill = document.getElementById('sig2-fill');
        s2Fill.style.width = `${(sigs.perplexity?.score || 0) * 100}%`;
        s2Fill.className = 'progress-fill ' + getFillClass(sigs.perplexity?.score);
        
        // Sig 3 (Linguistic)
        document.getElementById('sig3-label').textContent = sigs.linguistic?.label || '--';
        const s3Fill = document.getElementById('sig3-fill');
        s3Fill.style.width = `${(sigs.linguistic?.score || 0) * 100}%`;
        s3Fill.className = 'progress-fill ' + getFillClass(sigs.linguistic?.score);
        
        // Flags
        const flagsContainer = document.getElementById('flags-container');
        const flagsSection = document.getElementById('flags-section');
        flagsContainer.innerHTML = '';
        if (data.linguistic_flags && data.linguistic_flags.length > 0) {
            flagsSection.style.display = 'block';
            data.linguistic_flags.forEach(flag => {
                const el = document.createElement('span');
                el.className = 'flag';
                el.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${flag}`;
                flagsContainer.appendChild(el);
            });
        } else {
            flagsSection.style.display = 'none';
        }
        
        // Sentences
        const sCont = document.getElementById('sentence-container');
        sCont.innerHTML = '';
        if (data.sentences) {
            data.sentences.forEach(s => {
                const el = document.createElement('span');
                el.className = 'sentence-highlight';
                el.textContent = s.text + ' ';
                
                // Color scale: Green (low AI score) to Red (high AI score)
                const hue = 120 - (s.ai_score * 120); // 120 is green, 0 is red
                el.style.backgroundColor = `hsla(${hue}, 80%, 50%, 0.2)`;
                
                sCont.appendChild(el);
            });
        }
        
        // Radar Chart
        if (data.feature_highlights) {
            renderRadarChart(data.feature_highlights);
        }
    }
    
    function getFillClass(score) {
        if (!score && score !== 0) return '';
        if (score > 0.6) return 'fill-danger';
        if (score < 0.4) return 'fill-success';
        return 'fill-warning';
    }
    
    function renderRadarChart(features) {
        const ctx = document.getElementById('radarChart').getContext('2d');
        
        const labels = Object.keys(features).map(k => k.replace(/_/g, ' ').toUpperCase());
        const docValues = Object.values(features).map(f => f.value);
        const aiAvgs = Object.values(features).map(f => f.ai_avg);
        const humAvgs = Object.values(features).map(f => f.human_avg);
        
        // Normalize for radar (0-1 scale approx)
        const normalize = (val, idx) => {
            const f = Object.values(features)[idx];
            const max = Math.max(f.value, f.ai_avg, f.human_avg) * 1.2 || 1;
            return val / max;
        };
        
        const nDoc = docValues.map((v, i) => normalize(v, i));
        const nAi = aiAvgs.map((v, i) => normalize(v, i));
        const nHum = humAvgs.map((v, i) => normalize(v, i));
        
        if (radarChart) radarChart.destroy();
        
        radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'This Text',
                        data: nDoc,
                        backgroundColor: 'rgba(6, 182, 212, 0.4)',
                        borderColor: '#06b6d4',
                        borderWidth: 2,
                        pointBackgroundColor: '#06b6d4'
                    },
                    {
                        label: 'AI Average',
                        data: nAi,
                        backgroundColor: 'transparent',
                        borderColor: 'rgba(239, 68, 68, 0.5)',
                        borderDash: [5, 5],
                        borderWidth: 1
                    },
                    {
                        label: 'Human Average',
                        data: nHum,
                        backgroundColor: 'transparent',
                        borderColor: 'rgba(16, 185, 129, 0.5)',
                        borderDash: [5, 5],
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#94a3b8', font: { size: 10 } },
                        ticks: { display: false }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#f1f5f9', boxWidth: 12, padding: 10 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const idx = ctx.dataIndex;
                                const dsIdx = ctx.datasetIndex;
                                let val;
                                if (dsIdx === 0) val = docValues[idx];
                                else if (dsIdx === 1) val = aiAvgs[idx];
                                else val = humAvgs[idx];
                                return `${ctx.dataset.label}: ${val.toFixed(2)}`;
                            }
                        }
                    }
                }
            }
        });
    }
});
