from __future__ import annotations

import asyncio
import random
import re
import time

import numpy as np
import openai
import tiktoken
from prompts import (
    QA_ANSWER_SYSTEM_PROMPT,
    QA_REDUCE_PROMPT,
)
from sklearn.neighbors import KDTree  # type: ignore
from transformers import AutoTokenizer

from slingshot.schemas import OpenAIChatRequest, OpenAIChatResponse
from slingshot.sdk import SlingshotSDK

MODEL_FOR_EMBEDDING = "text-embedding-ada-002"

"""
This file houses the main logic for making requests to generate embeddings, and for determining the most relevant 
documents for a given question. It is used both by embed_documents.py and inference.py.
"""


def shorten_text(text: str, max_len: int, *, openai_model: str) -> str:
    """Shorten text to max_len tokens, using tiktoken encoding"""
    encoder = tiktoken.encoding_for_model(openai_model)
    encoded = encoder.encode(text)
    if len(encoded) < max_len:
        return text
    short_text_str_len = len(encoder.decode(encoded[:max_len]))
    return text[:short_text_str_len] + '...'


class TalkToDocs:
    def __init__(
        self,
        sdk: SlingshotSDK,
        documents: list[str],
        doc_embeddings: np.ndarray,
        openai_model: str = "gpt-3.5-turbo",
        # Note: OpenAI has some internal formatting tokens that count towards the token limit of 4096
        openai_model_token_limit: int = 4000,
        qa_reduce_prompt: str | None = None,
        qa_answer_system_prompt: str | None = None,
    ) -> None:
        assert len(doc_embeddings) == len(documents), (
            f"You should have the same number of documents and embeddings. Received {len(doc_embeddings)} embeddings "
            f"and {len(documents)} documents."
        )
        self.docs = documents
        self.doc_embeddings = doc_embeddings
        self.doc_embedding_kdtree = KDTree(doc_embeddings, leaf_size=2)

        self.openai_model = openai_model
        self.openai_model_token_limit = openai_model_token_limit
        self.tiktoken_encoder = tiktoken.encoding_for_model(openai_model)

        self.qa_reduce_prompt = qa_reduce_prompt or QA_REDUCE_PROMPT
        self.qa_answer_system_prompt = qa_answer_system_prompt or QA_ANSWER_SYSTEM_PROMPT

        # Slingshot stuff
        self.sdk = sdk
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    @staticmethod
    async def get_embedding(
        sdk: SlingshotSDK, text: str, num_retries: int = 5, retry_wait_period: float = 5
    ) -> np.ndarray:
        for retry in range(num_retries):
            try:
                response = await sdk.prompt_openai_embedding(text, model=MODEL_FOR_EMBEDDING)
                break
            except openai.error.APIError as e:
                print("Got an OpenAI API error:", e)
                print(f"Retrying in {retry_wait_period} seconds... ({retry + 1}/{num_retries})")
                await asyncio.sleep(retry_wait_period)
        else:
            raise RuntimeError(f"Failed to get embedding after {num_retries} retries.")
        embedding_data = response.data[0].embedding
        ndarray = np.array(embedding_data)
        return ndarray

    def get_num_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def shorten_multiple(self, texts: list[str], max_len: int = 3000) -> list[str]:
        """
        Keeps shortening the largest text from the list until the total length of all texts is <= max_len.
        """
        tokenized_texts = [self.tiktoken_encoder.encode(text) for text in texts]
        token_lengths = [len(tokenized_text) for tokenized_text in tokenized_texts]
        while sum(token_lengths) > max_len:
            # Get the first and second-largest texts
            sorted_token_lengths = np.argsort(token_lengths)
            largest_text_index = sorted_token_lengths[-1]
            second_largest_text_index = sorted_token_lengths[-2]
            # Shorten the largest text to the length of the second-largest text - 1
            token_lengths[largest_text_index] = token_lengths[second_largest_text_index] - 1
        print(f"Shortened text lengths: {token_lengths}; total length: {sum(token_lengths)}")
        return [
            self.tiktoken_encoder.decode(tokenized_text[:token_length])
            for tokenized_text, token_length in zip(tokenized_texts, token_lengths)
        ]

    async def get_relevant_documents(
        self, query: str, num_return_docs: int = 10
    ) -> list[str]:
        num_return_docs = min(num_return_docs, len(self.docs))
        query_embedding = np.array(await self.get_embedding(self.sdk, query)).reshape(1, -1)
        distances, indices = self.doc_embedding_kdtree.query(query_embedding, k=num_return_docs)
        return [self.docs[i] for i in indices.reshape(-1)]

    async def get_answer_based_on_docs(
        self, question: str, excerpts: list[str], max_output_tokens: int = 1000
    ) -> tuple[str, list[int]]:
        num_excerpts = len(excerpts)
        used_indices = {6, 9, 33, 46, 17, 18, 12, 13, 11, 4, 7}
        unused_indices = [num for num in range(num_excerpts + len(used_indices)) if num not in used_indices]
        random.seed(42)
        random.shuffle(unused_indices)
        excerpt_index_mapping = {
            unused_index: excerpt_index for excerpt_index, unused_index in enumerate(unused_indices)
        }

        # Shorten excerpts to fit in the query
        tokenized_question = self.tiktoken_encoder.encode(question)
        tokenized_prompt = self.tiktoken_encoder.encode(self.qa_reduce_prompt.format(question="", excerpts=""))
        system_prompt_length = len(self.tiktoken_encoder.encode(self.qa_answer_system_prompt))
        excerpt_formatting_length = len(self.tiktoken_encoder.encode(self.format_excerpt("", 10))) * num_excerpts
        max_excerpts_len = (
            self.openai_model_token_limit
            - len(tokenized_question)
            - len(tokenized_prompt)
            - max_output_tokens
            - system_prompt_length
            - excerpt_formatting_length
        )
        print(f"Excerpt length limit: {max_excerpts_len}")
        excerpts = self.shorten_multiple(excerpts, max_len=max_excerpts_len)
        excerpts_str = "\n".join(self.format_excerpt(excerpt, i) for i, excerpt in zip(unused_indices, excerpts))
        excerpts_str_token_length = len(self.tiktoken_encoder.encode(excerpts_str))
        prompt = self.qa_reduce_prompt.format(question=question, excerpts=excerpts_str)
        prompt_length = len(self.tiktoken_encoder.encode(prompt))
        body: list[dict[str, str]] = [
            {"role": "system", "content": self.qa_answer_system_prompt},
            {"role": "user", "content": prompt},
        ]
        print(
            f"Final answer prompt: template length = {len(tokenized_prompt)}; question length = "
            f"{len(tokenized_question)}; excerpts length = {excerpts_str_token_length}; system prompt length = "
            f"{system_prompt_length}; total length = {prompt_length + system_prompt_length}"
        )
        answer = await self.get_completion(body)
        excerpt_indices = self.extract_answer_sources(answer, excerpt_index_mapping)

        # Find the index where "sources:" starts and remove everything after that
        answer, _, __ = re.split(r"(sources?):", answer, flags=re.IGNORECASE)
        return answer, excerpt_indices

    async def get_answer(self, question: str) -> tuple[str, list[str]]:
        print("Getting relevant documents...")
        start = time.time()
        relevant_docs = await self.get_relevant_documents(question)
        end = time.time()
        print(f"Got {len(relevant_docs)} relevant documents in {end - start} seconds.")
        print("Getting excerpts from relevant documents...")
        start = time.time()
        print(f"Relevant docs: {relevant_docs}")
        answer, sources_indices = await self.get_answer_based_on_docs(question, relevant_docs)
        sources = [relevant_docs[i] for i in sources_indices]
        end = time.time()
        print(f"Got answer in {end - start} seconds.")
        return answer, sources

    async def get_completion(self, body: list[dict[str, str]], max_tokens: int = 1000) -> str:
        print("=" * 80)
        print(f"Getting completion for prompt: {body}")
        print("=" * 80)
        response = await self.sdk.prompt_openai_chat(
            OpenAIChatRequest(
                model="gpt-3.5-turbo",
                messages=body,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0,
            )
        )
        if response.choices[0].finish_reason != "stop":
            # Check if reason for stopping was the token limit
            # We should perhaps monitor this to know if the `max_tokens` parameter is adjusted well for our use-case
            print(
                "WARNING! Model didn't stop generating text on its own. It is possible you have set max_tokens too low."
            )
        return response.choices[0].message['content']

    @staticmethod
    def format_excerpt(excerpt: str, source: int) -> str:
        return f"Content: {excerpt}\nSource: {source}-pl"

    @staticmethod
    def extract_answer_sources(answer: str, sources_mapping: dict[int, int]) -> list[int]:
        if "sources:" not in answer.lower():
            return []
        sources_str = answer.lower().split("sources:")[-1]
        regex_pattern = r"(\d+)-pl"
        source_indices = [int(match.group(1)) for match in re.finditer(regex_pattern, sources_str)]
        return [sources_mapping[source_index] for source_index in source_indices]
