import modal
import os
from mistralai import Mistral

app = modal.App("fries-coder")


@app.function(
    secret=modal.Secret.from_name("mistral-api-key"),
    required_keys=["MISTRAL_API_KEY"],
)
def generate_code(prompt: str, suffix: str) -> str:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.fim.complete(
        model="codestral-latest",
        prompt=prompt,
        suffix=suffix,
        temperature=0.7,
        top_p=1,
    )
    return response.choices[0].message.content


import modal

# Define the custom Modal image
image = modal.Image.debian_slim().pip_install(
    "llama-index",
    "llama-index-llms-mistralai",
    "mistralai",
    "nest_asyncio",
)

# Define the Modal function
stub = modal.Stub("llamaindex-mistralai-modal")


@stub.function(image=image)
def run_llamaindex_with_mistralai(prompt: str):
    import os
    from llama_index.llms.mistralai import MistralAI
    from llama_index.embeddings.mistralai import MistralAIEmbedding
    from llama_index import VectorStoreIndex

    # Set up MistralAI
    os.environ["MISTRAL_API_KEY"] = "your-mistral-api-key"
    llm = MistralAI(model="open-mixtral-8x22b", temperature=0.1)
    embed_model = MistralAIEmbedding(model_name="mistral-embed")

    # Initialize LlamaIndex
    index = VectorStoreIndex.from_documents(["Sample document for testing"])
    response = index.query(prompt)

    return response


# Run the function
if __name__ == "__main__":
    with stub.run():
        print(
            run_llamaindex_with_mistralai.call(
                "What is the summary of the document?"
            )
        )
