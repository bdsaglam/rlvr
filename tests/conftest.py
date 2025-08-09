import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env_file():
    assert load_dotenv(), "Failed to load .env file"
