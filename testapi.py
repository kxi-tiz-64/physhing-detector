import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "email": "Dear user, your PayPal account has been flagged. Please verify your identity."
}

response = requests.post(url, json=payload)
print(response.json())
