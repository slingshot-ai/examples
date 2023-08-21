from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
from qa_llm import TalkToDocs

from slingshot import InferenceModel, Prediction
from slingshot.sdk import SlingshotSDK

"""
This file is run by our deployment to handle incoming inference requests.

load() - prepares our TalkToDocs object in qa_llm.py, which will houses the logic for finding relevant docs, and making
the eventual augmented request to openai.

predict() - passes on an inference request to our TalkToDocs object, and handles batching these requests appropriately. 
"""


class DocQA(InferenceModel):
    async def load(self) -> bool:
        """
        Slingshot will call this method to load the model.

        Implementation example:
            self.model = torch.load("/mnt/model/model.pt")
        """
        sdk = SlingshotSDK()

        # Load embeddings
        self.embeddings_path = Path("/mnt/embeddings/embeddings.npy")
        doc_embeddings = np.load(str(self.embeddings_path))
        documents_path = Path("/mnt/documents")
        documents_path = documents_path / os.listdir(documents_path)[0]
        with open(documents_path) as f:
            documents = json.load(f)
        self.model = TalkToDocs(sdk, documents, doc_embeddings)
        self.ready = True
        return True

    async def predict(self, examples: list[bytes]) -> Prediction | list[Prediction]:
        """
        Slingshot will call this method to make predictions, passing in the raw request bytes and returns a dictionary.
        For text inputs, the bytes will be the UTF-8 encoded string.

        If the model is not batched, the input will be a list with a single element and the output should be a single
        dictionary as the prediction response. Otherwise, the input will be a list of examples and the output should be
        a list of dictionaries with the same length and order as the input.

        Implementation example:
            example_text = examples[0].decode("utf-8")
            return self.model(example_text)
        """

        async def return_result_with_index(task, index):
            try:
                result = await task
            except Exception as e:
                return (str(e), []), index
            return result, index

        inference_tasks = []
        print(f"Received {len(examples)} requests to respond")
        for i, question in enumerate(examples):
            question = question.decode("utf-8")
            inference_tasks.append(return_result_with_index(self.model.get_answer(question), i))

        predictions = {}
        for task_coro in asyncio.as_completed(inference_tasks):
            (answer, sources), task_index = await task_coro
            sources = [source.strip().strip("\"") for source in sources]
            predictions[task_index] = {"answer": answer.strip().strip("\""), "sources": sources}
        ordered_predictions = [predictions[i] for i in range(len(predictions))]
        return {"predictions": ordered_predictions}


if __name__ == "__main__":
    model = DocQA()
    model.start()
