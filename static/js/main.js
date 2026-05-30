document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------------------
    // 1. DOM Elements
    // ----------------------------------------------------------------
    const btnSingle = document.getElementById('btn-single');
    const btnBatch = document.getElementById('btn-batch');
    const panelSingle = document.getElementById('panel-single');
    const panelBatch = document.getElementById('panel-batch');
    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');
    
    // Sliders
    const tenureSlider = document.getElementById('tenure');
    const tenureVal = document.getElementById('tenure-val');
    const monthlySlider = document.getElementById('MonthlyCharges');
    const monthlyVal = document.getElementById('monthly-val');
    const totalChargesInput = document.getElementById('TotalCharges');
    
    // Form and Results (Single)
    const singleForm = document.getElementById('single-predict-form');
    const riskGauge = document.getElementById('risk-gauge');
    const riskScore = document.getElementById('risk-score');
    const riskBadge = document.getElementById('risk-badge');
    const reasonsList = document.getElementById('risk-reasons-list');
    
    // Upload & Batch Elements
    const batchForm = document.getElementById('batch-upload-form');
    const fileInput = document.getElementById('batch-file');
    const dropZone = document.getElementById('csv-drop-zone');
    const fileInfo = document.getElementById('selected-file-info');
    const selectedFilename = document.getElementById('selected-filename');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const loader = document.getElementById('processing-loader');
    const batchResults = document.getElementById('batch-results-panel');
    
    // Batch Stats
    const statTotal = document.getElementById('stat-total');
    const statChurned = document.getElementById('stat-churned');
    const statRetained = document.getElementById('stat-retained');
    const statRate = document.getElementById('stat-rate');
    const downloadLink = document.getElementById('download-predictions-link');
    
    let churnDonutChart = null;

    // ----------------------------------------------------------------
    // 2. Tab Navigation
    // ----------------------------------------------------------------
    btnSingle.addEventListener('click', () => {
        btnSingle.classList.add('active');
        btnBatch.classList.remove('active');
        panelSingle.classList.add('active');
        panelBatch.classList.remove('active');
        pageTitle.innerText = "Predictive Retention Insights";
        pageSubtitle.innerText = "Analyze churn risk profile for individual customer accounts";
    });

    btnBatch.addEventListener('click', () => {
        btnBatch.classList.add('active');
        btnSingle.classList.remove('active');
        panelBatch.classList.add('active');
        panelSingle.classList.remove('active');
        pageTitle.innerText = "Batch Churn Analytics";
        pageSubtitle.innerText = "Upload customer datasets to analyze portfolio attrition risks";
    });

    // ----------------------------------------------------------------
    // 3. Dynamic Sliders
    // ----------------------------------------------------------------
    function updateValAndEstimatedTotal() {
        const tenure = parseInt(tenureSlider.value);
        const monthly = parseFloat(monthlySlider.value);
        
        tenureVal.innerText = tenure;
        monthlyVal.innerText = monthly.toFixed(2);
        
        // Auto-calculate estimated TotalCharges (MonthlyCharges * tenure)
        const estimatedTotal = monthly * tenure;
        totalChargesInput.value = estimatedTotal.toFixed(2);
    }

    tenureSlider.addEventListener('input', updateValAndEstimatedTotal);
    monthlySlider.addEventListener('input', updateValAndEstimatedTotal);

    // ----------------------------------------------------------------
    // 4. SVG Progress Ring Gauge Logic
    // ----------------------------------------------------------------
    const radius = riskGauge.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    
    riskGauge.style.strokeDasharray = `${circumference} ${circumference}`;
    riskGauge.style.strokeDashoffset = circumference;

    function setRiskGaugeValue(percent) {
        const offset = circumference - (percent / 100 * circumference);
        riskGauge.style.strokeDashoffset = offset;
    }

    // ----------------------------------------------------------------
    // 5. Single Predict Form Submission
    // ----------------------------------------------------------------
    singleForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Build payload dict
        const formData = new FormData(singleForm);
        const payload = {};
        
        // Defaults
        payload['SeniorCitizen'] = 0;
        payload['Partner'] = 'No';
        payload['Dependents'] = 'No';
        payload['PhoneService'] = 'No';
        payload['MultipleLines'] = 'No';
        payload['PaperlessBilling'] = 'No';
        
        formData.forEach((value, key) => {
            if (key === 'SeniorCitizen') {
                payload[key] = parseInt(value);
            } else if (key === 'tenure') {
                payload[key] = parseInt(value);
            } else if (key === 'MonthlyCharges' || key === 'TotalCharges') {
                payload[key] = parseFloat(value);
            } else {
                payload[key] = value;
            }
        });

        // Handle unchecked checkboxes
        const checkboxes = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'PaperlessBilling'];
        checkboxes.forEach(c => {
            if (!formData.has(c)) {
                if (c === 'SeniorCitizen') payload[c] = 0;
                else payload[c] = 'No';
            }
        });

        try {
            // Show predicting state
            riskScore.innerText = "---";
            riskBadge.className = "risk-badge";
            riskBadge.innerText = "Analyzing...";
            reasonsList.innerHTML = "<li>Evaluating model features...</li>";
            
            const response = await fetch('/predict_single', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            if (result.error) {
                alert(result.error);
                return;
            }

            const probPct = result.churn_probability * 100;
            
            // Animate Gauge & Text
            setRiskGaugeValue(probPct);
            riskScore.innerText = `${probPct.toFixed(1)}%`;
            
            // Set Risk Level styling
            riskBadge.className = `risk-badge ${result.risk_level.toLowerCase()}`;
            riskBadge.innerText = `${result.risk_level} Risk`;
            
            // Set dynamic SVG gradient colors based on risk
            const stop1 = document.getElementById('gradient').children[0];
            const stop2 = document.getElementById('gradient').children[1];
            if (result.risk_level === 'High') {
                stop1.setAttribute('stop-color', '#ff0844');
                stop2.setAttribute('stop-color', '#ffb199');
            } else if (result.risk_level === 'Medium') {
                stop1.setAttribute('stop-color', '#f857a6');
                stop2.setAttribute('stop-color', '#ff5858');
            } else {
                stop1.setAttribute('stop-color', '#11998e');
                stop2.setAttribute('stop-color', '#38ef7d');
            }

            // Populate explanations
            reasonsList.innerHTML = "";
            result.reasons.forEach(reason => {
                const li = document.createElement('li');
                li.innerText = reason;
                reasonsList.appendChild(li);
            });

        } catch (err) {
            console.error(err);
            alert("Connection error while communicating with predictions model.");
        }
    });

    // ----------------------------------------------------------------
    // 6. CSV Batch Upload & Drag/Drop
    // ----------------------------------------------------------------
    
    // Drag/Drop visual triggers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped file
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    // Browse click trigger
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.name.endsWith('.csv')) {
            alert("Only CSV files are supported.");
            return;
        }
        
        // Save file reference to input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;

        // Show file details panel
        selectedFilename.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        dropZone.style.display = 'none';
        fileInfo.classList.remove('selected-file-hidden');
    }

    // Reset file selection
    removeFileBtn.addEventListener('click', () => {
        fileInput.value = '';
        fileInfo.classList.add('selected-file-hidden');
        dropZone.style.display = 'flex';
        batchResults.classList.add('inactive');
    });

    // ----------------------------------------------------------------
    // 7. Batch Predict Form Submission
    // ----------------------------------------------------------------
    batchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            alert("Please select or drop a CSV file to process.");
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            loader.classList.remove('hidden');
            batchResults.classList.add('inactive');

            const response = await fetch('/predict_batch', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            loader.classList.add('hidden');

            if (result.error) {
                alert(result.error);
                return;
            }

            // Populate stats
            statTotal.innerText = result.total.toLocaleString();
            statChurned.innerText = result.churned.toLocaleString();
            statRetained.innerText = result.retained.toLocaleString();
            statRate.innerText = `${result.churn_rate}%`;
            
            // Setup download link
            downloadLink.href = result.download_url;

            // Render Donut Chart
            renderChurnDonutChart(result.retained, result.churned);

            // Display Results
            batchResults.classList.remove('inactive');

        } catch (err) {
            loader.classList.add('hidden');
            console.error(err);
            alert("Batch prediction task failed. Ensure dataset has correct columns.");
        }
    });

    // Chart.js render donut helper
    function renderChurnDonutChart(retained, churned) {
        const ctx = document.getElementById('churnChart').getContext('2d');
        
        if (churnDonutChart) {
            churnDonutChart.destroy();
        }

        churnDonutChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Retained', 'Churned'],
                datasets: [{
                    data: [retained, churned],
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: '#171d27',
                    borderWidth: 3,
                    hoverOffset: 6
                }]
            },
            options: {
                cutout: '72%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed !== null) {
                                    label += context.parsed.toLocaleString();
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }

});
