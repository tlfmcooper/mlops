import os
import sys

# Ensure the parent directory is added to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask_app import app
import pytest
import json


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    input_data = {"age": 40, "estimated_salary": 522222}

    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    response_data = json.loads(response.data)
    print(response_data)
    assert "prediction" in response_data
    assert response_data["prediction"] in [
        "Not likely to purchase",
        "Likely to purchase",
    ]
