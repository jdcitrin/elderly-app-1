import requests

# test health check
print("testing backend")
response = requests.get('http://localhost:3000/')
print("health check", response.json())

# test prediction
with open('../test/mangotest.webp', 'rb') as f:
    files = {'image': f}
    data = {'category': 'fruit'}
    response = requests.post('http://localhost:3000/predict', files=files, data=data)
    print("\nprediction:", response.json())