document.getElementById('loanForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);
    const data = {};

    formData.forEach((value, key) => {
        if (["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"].includes(key)) {
            data[key] = parseFloat(value);
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

        const result = await res.json();

        if (result.prediction) {
            resultBox.innerHTML = `<h2>Loan Prediction: ${result.prediction}</h2>`;
        } else {
            resultBox.innerHTML = `<h2>Error: ${result.error}</h2>`;
        }
    } catch (err) {
        resultBox.innerHTML = `<h2>Request failed</h2>`;
    }
});