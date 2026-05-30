document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------------------
    // 1. DOM Elements
    // ----------------------------------------------------------------
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

    // ----------------------------------------------------------------
    // 2. Dynamic Sliders
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

    // Initial load sync
    updateValAndEstimatedTotal();

    // ----------------------------------------------------------------
    // 3. SVG Progress Ring Gauge Logic
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
    // 4. Single Predict Form Submission
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

});
