import requests
response = requests.get("http://10.28.228.79:30390/v1/models", timeout=5)
print(response.status_code)