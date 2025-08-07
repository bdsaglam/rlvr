import logging
import random

from datasets import load_dataset
from locust import HttpUser, between, events, task

from verifiers.datasets.musique import preprocess_dataset

dataset = preprocess_dataset(
    load_dataset("bdsaglam/musique", "answerable", split="validation[:100]")
)


class RerankUser(HttpUser):
    # Wait between 1 to 2 seconds between tasks
    wait_time = between(1, 2)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @task
    def rerank_documents(self):
        # Randomly select an example from dataset
        i = random.choice(list(range(len(dataset))))
        example = dataset[i]

        # Get question from prompt
        query = example["prompt"][0]["content"]

        # Get documents from docs field
        docs = [doc["text"] for doc in example["docs"]]

        payload = {
            "query": query,
            "documents": docs,
            "top_n": 1,
            "return_documents": False,
            # "model": "bm25",
            "model": "tei",
        }

        with self.client.post(
            "/rerank",
            json=payload,
            catch_response=True,
            timeout=30.0,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Request failed with status code: {response.status_code}"
                )


# Add event listeners for test statistics


@events.quitting.add_listener
def _(environment, **kw):
    if environment.stats.total.fail_ratio > 0.01:
        logging.error("Test failed due to failure ratio > 1%")
        environment.process_exit_code = 1
    elif environment.stats.total.avg_response_time > 10000:
        logging.error(
            "Test failed due to average response time ratio > 1000 ms"
        )
        environment.process_exit_code = 1
    else:
        environment.process_exit_code = 0
