document.getElementById("analyzeBtn").addEventListener("click", async () => {
    const text = document.getElementById("inputText").value.trim();

    if (!text) {
        alert("Please enter a sentence!");
        return;
    }

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        document.getElementById("sentimentLabel").innerText =
            "Predicted Sentiment: " + data.sentiment;
        document.getElementById("confidenceScore").innerText =
            "Confidence: " + (data.confidence * 100).toFixed(2) + "%";
    } catch (error) {
        console.error("Error:", error);
        alert("Prediction failed. Check your backend server.");
    }
});