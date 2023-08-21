# Document Q&A

This example [Slingshot](https://www.slingshot.xyz/) project walks through how to combine an existing set of documents
with GPT 3.5 turbo, and place the result behind a front-end app that allows you to query the augmented model.

We're using the [essays of Paul Graham ](http://www.paulgraham.com/) in this example, but the modular nature of the
project allows you to swap out the essays for any set of documents. You could easily modify this example to create a
GPT 3.5-powered chatbot augmented with all of your internal company knowledge, or a chatbot that has knowledge of the
most recent publications of leading medical journals.

This project is a perfect blueprint to demonstrate how you can combine existing LLMs with outside information to create
a more specialized and useful tool.

## Project setup

1. Clone this repo
2. In your project directory, initialize a new Slingshot project:

```bash
$ slingshot init
```

Create a new project when prompted.

Finally, run `slingshot push` to push local files to your Slingshot project

## Download Graham's essays

First, let's prepare our retrieval dataset. For this use case, we'll scrape Paul Graham's essays from his blog. We've
already written a script for this, which you can run via the Runs page, or from the commandline with:

```bash
slingshot run download-essays
```

This will create a new artifact containing the parsed text of Graham's essays.

## Generate Embeddings

As part of our pipeline, we use document embeddings to retrieve initial candidates for relevant essays regarding the
asked question.

Since the embeddings are static for a given essay, we can precompute them and save them into an artifact as well. In
this example project, we've also provided code for this.

For this Run, we're going to use OpenAI to compute document embeddings for us, so be sure to set you OpenAI API key as a
secret. This can be done vie the "Secrets" tab or via CLI:

```bash
$ slingshot secret openai
```

Once that's done, go to the Runs page, but this time start the `embed-documents` Run. After this you should now have
another artifact with all the precomputed embeddings for each essay.

## Start a Deployment

Once the download and preprocessing steps are done, you can start `document-qa-deployment` deployment. This will spin
up a Deployment that takes in questions as strings and responds with replies as string as well. You can test it right
away by going to the "Test tab" in the Deployment page and submitting a question.

This is however not a very friendly UI. Let's create a nicer interface!

## Start an App

If you want to have a better experience asking questions about Paul Graham's essays, or get a nice interface to share
with friends, you can use a Slingshot App. We've provided a very simple one for this example that uses Gradio to
interact with the deployment you started on the previous section.

Simply go to the Apps page and start the one called `web`. Once it finishes starting up (which should take a few seconds
at most), head on over to the provided URL and start asking questions!

The link is publicly accessible, so feel free to share it with friends or post it online.
