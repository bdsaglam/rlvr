import logging
from functools import cache

from openai import OpenAI

log = logging.getLogger(__name__)


@cache
def get_default_model(client: OpenAI = None):
    openai_client = client or OpenAI()
    models_response = openai_client.models.list()
    available_models = [item.id for item in models_response.data if item.object == "model"]
    log.info(f"Available models: {available_models}")
    model = available_models[0]
    return model
