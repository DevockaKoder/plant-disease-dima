from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

healthy_leaf = open('../healthy.jpg', "rb")

def test_read_main():
    response = client.get("/test")
    assert response.status_code == 200
    assert "Local server is started"


def test_predict_positive():
    response = client.post("/predict", files={"file": ("healthy", healthy_leaf, "image/jpeg")})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['class'] == 'Healthy'


def test_predict_negative():
    response = client.post("/predict", files={"file": ("healthy", healthy_leaf, "image/jpeg")})
    json_data = response.json()
    assert response.status_code == 200
    #Negative scenario
    assert json_data['class'] == 'Bad'