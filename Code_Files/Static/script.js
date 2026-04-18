document.getElementById('loanForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const data = {};

    formData.forEach((value, key) => {
        if (["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"].includes(key)) {
            data[key] = value === '' ? null : parseFloat(value);
        } else {
            data[key] = value;
        }
    });

    const resultBox = document.getElementById('resultBox');
    resultBox.innerHTML = 'Predicting...';

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const json = await res.json();

        if (json.error) {
            resultBox.innerHTML = `<h2>Error: ${json.error}</h2>`;
            return;
        }

        const probPercent = Math.round((json.risk_probability || 0) * 100);
        const level = json.risk_level || 'Unknown';
        const colorClass = level.toLowerCase();

        resultBox.innerHTML = `
            <div class="result-card">
                <h2>Decision: ${json.prediction}</h2>
                <p><strong>Recommended action:</strong> ${json.recommended_action}</p>
                <p><strong>Risk probability:</strong> ${probPercent}%</p>
                <p><strong>Risk level:</strong> <span class="risk ${colorClass}">${level}</span></p>
                <h3>Top risk drivers</h3>
                <ul>${(json.risk_drivers || []).map(d => `<li>${d}</li>`).join('')}</ul>
            </div>
        `;

    } catch (err) {
        resultBox.innerHTML = `<h2>Request failed</h2>`;
    }
});