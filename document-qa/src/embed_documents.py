import asyncio
import json
import os
from pathlib import Path

import numpy as np
from qa_llm import MODEL_FOR_EMBEDDING, TalkToDocs, shorten_text
from tqdm.auto import tqdm

from slingshot.sdk import SlingshotSDK

"""
Script used to generate embeddings for the mounted documents file - in our base example these are essays by Paul Graham. # TODO: Describe schema of file
This file contains logic to check for and skip docs we've already created embeddings for, then pass on the request to 
generate the new embeddings to qa_llm.py.
"""


async def get_embedding(text: str, *, sdk: SlingshotSDK) -> np.ndarray:
    try:
        await asyncio.sleep(1)
        return await TalkToDocs.get_embedding(sdk, shorten_text(text, 8189, openai_model=MODEL_FOR_EMBEDDING))
    except RuntimeError as e:
        print(f"Failed to get embedding, filling it with zeros.") # This isn't cool at all
        print(e)
        return np.zeros(1536) # Why is this 1536?


async def embed_documents(
    doc_texts: list[str], *, precomputed_embeddings: np.ndarray | None = None, sdk: SlingshotSDK
) -> np.ndarray:
    if precomputed_embeddings is None:
        precomputed_embeddings = np.zeros((len(doc_texts), 1536))

    new_embeddings = precomputed_embeddings.sum(axis=1) == 0
    num_new_embeddings = new_embeddings.sum()
    print(f"Actually embedding {num_new_embeddings} of {len(doc_texts)} documents. (the rest is cached)")
    print(f"New embeddings indices: {np.where(new_embeddings)}")

    async def return_embedding(embedding: np.ndarray) -> np.ndarray: # TODO: Why is this here?
        return embedding

    embeddings = await tqdm.gather(
        *[
            get_embedding(text, sdk=sdk) if embedding.sum() == 0 else return_embedding(embedding)
            for text, embedding in zip(doc_texts, precomputed_embeddings)
        ]
    )
    return np.array(embeddings)


if __name__ == "__main__":
    sdk = SlingshotSDK()
    docs_folder = Path("/mnt/documents")
    docs_path = docs_folder / os.listdir(docs_folder)[0]

    with open(docs_path) as f: # TODO: Change to pandas; be clear on schema
        docs = json.load(f)

    # Check for pre-computed embeddings, if we're re-computing for some new data
    if os.path.exists("/mnt/embeddings/embeddings.npy"):
        already_computed_embeddings = np.load("/mnt/embeddings/embeddings.npy")
    else:
        already_computed_embeddings = None

    computed_embeddings = asyncio.run(
        embed_documents(docs, precomputed_embeddings=already_computed_embeddings, sdk=sdk)
    )
    np.save("/mnt/embeddings/embeddings.npy", computed_embeddings)
