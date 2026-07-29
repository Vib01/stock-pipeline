import os, requests
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("GITHUB_TOKEN")
repo  = os.getenv("GITHUB_REPO")

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json",
}

url = f"https://api.github.com/repos/{repo}/contents/ranking.json"
r = requests.get(url, headers=headers)
print(f"GET {r.status_code}: {r.json()}")

import base64, json
body = {"message": "test", "content": base64.b64encode(b"{}").decode()}
r2 = requests.put(url, headers=headers, json=body)
print(f"PUT {r2.status_code}: {r2.json()}")
