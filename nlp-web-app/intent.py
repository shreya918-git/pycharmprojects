import requests
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
headers = {
    
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

