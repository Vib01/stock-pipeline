import os, requests
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("GITHUB_TOKEN")
repo  = os.getenv("GITHUB_REPO")

r = requests.get("https://api.github.com/user", headers={"Authorization": f"token {token}"})
print(f"Token status: {r.status_code}, login: {r.json().get('login')}")

r2 = requests.get(f"https://api.github.com/repos/{repo}", headers={"Authorization": f"token {token}"})
print(f"Repo status: {r2.status_code}, data: {r2.json().get('full_name', r2.json().get('message'))}")
