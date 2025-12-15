document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const modal = document.getElementById('resultModal');
    const closeModal = document.querySelector('.close');
    const resultTitle = document.getElementById('resultTitle');
    const resultText = document.getElementById('resultText');
    const resultIcon = document.getElementById('resultIcon');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show Loading
        setLoading(true);

        // Gather Data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                showResult(result.prediction);
            } else {
                showResult("Error", result.error || "An error occurred");
            }
        } catch (error) {
            console.error('Error:', error);
            showResult("Error", "Failed to connect to server.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            btnText.textContent = "Processing...";
            btnSpinner.style.display = 'block';
            predictBtn.disabled = true;
        } else {
            btnText.textContent = "Predict Eligibility";
            btnSpinner.style.display = 'none';
            predictBtn.disabled = false;
        }
    }

    function showResult(prediction, errorMessage) {
        modal.style.display = "flex";
        // Force reflow for transition
        void modal.offsetWidth;
        modal.classList.add('show');

        if (errorMessage) {
            resultTitle.textContent = "Error";
            resultTitle.className = "rejected";
            resultText.textContent = errorMessage;
            resultIcon.innerHTML = "⚠️";
            resultIcon.className = "icon rejected";
            return;
        }

        if (prediction === "Approved" || prediction === "Y") {
            resultTitle.textContent = "Congratulations!";
            resultTitle.className = "approved";
            resultText.textContent = "Based on the details provided, the loan application is likely to be APPROVED.";
            resultIcon.innerHTML = "✅";
            resultIcon.className = "icon approved";
        } else {
            resultTitle.textContent = "Application Rejected";
            resultTitle.className = "rejected";
            resultText.textContent = "Based on the details provided, the loan application is likely to be REJECTED.";
            resultIcon.innerHTML = "❌";
            resultIcon.className = "icon rejected";
        }
    }

    // Modal Close Logic
    closeModal.onclick = () => {
        closeModalFunc();
    }

    window.onclick = (event) => {
        if (event.target == modal) {
            closeModalFunc();
        }
    }

    function closeModalFunc() {
        modal.classList.remove('show');
        setTimeout(() => {
            modal.style.display = "none";
        }, 300);
    }
});
