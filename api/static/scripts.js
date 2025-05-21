async function detectPhishing() {
  const input = document.getElementById("emailInput").value;
  const resultDiv = document.getElementById("result");

  if (!input.trim()) {
    resultDiv.innerHTML = `<span class="text-yellow-500 dark:text-yellow-300">⚠️ Please enter some text!</span>`;
    return;
  }

  resultDiv.innerText = "Analyzing...";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {  // Full URL in case you're running locally
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ email: input })  // Matches Flask route expecting 'email'
    });

    const data = await response.json();

    if (data.prediction === "Phishing") {
      resultDiv.innerHTML = `<span class="text-red-500 dark:text-red-400">⚠️ Phishing Detected</span>`;
    } else if (data.prediction === "Legitimate") {
      resultDiv.innerHTML = `<span class="text-green-600 dark:text-green-400">✅ Legitimate Email</span>`;
    } else {
      resultDiv.innerHTML = `<span class="text-yellow-500">Unexpected response: ${JSON.stringify(data)}</span>`;
    }
  } catch (error) {
    console.error(error);
    resultDiv.innerHTML = `<span class="text-red-500">❌ Error connecting to the server.</span>`;
  }
}
