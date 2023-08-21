from __future__ import annotations

import gradio as gr
from fastapi import FastAPI

from slingshot.sdk.slingshot_sdk import SlingshotSDK

app = FastAPI()
sdk = SlingshotSDK()
DEPLOYMENT_NAME = "document-qa-deployment"  # Substitute your deployment name here if you change it

"""
Front-end app built with Gradio that allows you to interact with the talk-to-docs-deployment through a nice UI. 
"""


async def answer_question(question: str) -> tuple[str, dict[str, str]]:
    print("Received question")
    print(f"Question: {question}")

    # Convert the question string to bytes
    question_bytes = question.encode("utf-8")

    response = await sdk.predict(DEPLOYMENT_NAME, question_bytes)
    if "predictions" not in response:
        return "Unable to process request"
    print(f"Prediction response: {response}")
    answer = response["predictions"][0]["answer"]
    return answer


# Create and run the Gradio app
print("Starting app")
iface = gr.Interface(answer_question, inputs="text", outputs=["text"])
app = gr.mount_gradio_app(app, iface, path="/")
