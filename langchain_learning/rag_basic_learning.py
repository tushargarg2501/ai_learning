import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredHTMLLoader,
)
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

INDEX_NAME = "coffeeindex"
NAMESPACE = "ragdemo"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("Set PINECONE_API_KEY in langchain_learning/.env.")

pine_cone = Pinecone(api_key=pinecone_api_key)

pages_dir = PROJECT_DIR / "coffee_pages"

loader = DirectoryLoader(
    str(pages_dir),
    glob="**/*.html",
    loader_cls=UnstructuredHTMLLoader,
)
docs = loader.load()
print("Loaded:", len(docs))
for document in docs:
    source = str(document.metadata.get("source", ""))
    document.metadata["category"] = Path(source).stem.replace("-", "_")

# Split the documents into chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
)
chunks = text_splitter.split_documents(docs)
print("Split into chunks:", len(chunks))
for position, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {position}")
    print(f"Metadata: {chunk.metadata}")
    print(f"Text:\n{chunk.page_content}")

# Define the embeddings model.
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if not chunks:
    raise ValueError(f"No document chunks found in {pages_dir}.")

embedding_dimension = len(embeddings_model.embed_query("dimension check"))
if embedding_dimension == 0:
    raise ValueError("The embedding model returned an empty vector.")

if not pine_cone.has_index(INDEX_NAME):
    pine_cone.create_index(
        name=INDEX_NAME,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

vector_ids = []
for position, chunk in enumerate(chunks):
    source = str(chunk.metadata.get("source", ""))
    identity = f"{source}:{position}:{chunk.page_content}"
    vector_ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())

vectorstore_from_docs = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    ids=vector_ids,
    index_name=INDEX_NAME,
    namespace=NAMESPACE,
    text_key="text",
)
print(
    f"Upserted {len(chunks)} vectors to {INDEX_NAME!r}/{NAMESPACE!r} "
    "using LangChain."
)

query_text = "How is cold brew coffee made?"
multi_query_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_ctx=8192,
)
self_query_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_ctx=8192,
    format="json",
)

base_retriever = vectorstore_from_docs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=multi_query_llm,
    include_original=True,
)
multi_query_docs = multi_query_retriever.invoke(query_text)

print(f"\nMulti-query results for {query_text!r}:")
for document in multi_query_docs:
    print(
        f"category={document.metadata.get('category', '')} "
        f"source={document.metadata.get('source', '')}"
    )
    print(document.page_content)

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=self_query_llm,
    vectorstore=vectorstore_from_docs,
    document_contents="Coffee brewing guides describing methods and preparation.",
    metadata_field_info=[
        AttributeInfo(
            name="category",
            description=(
                "The coffee brewing guide category. Values are cold_brew, espresso, "
                "or pour_over."
            ),
            type="string",
        ),
        AttributeInfo(
            name="source",
            description="The full local path of the HTML source file.",
            type="string",
        ),
    ],
    structured_query_translator=PineconeTranslator(),
    search_kwargs={"k": 3},
)
self_query_docs = self_query_retriever.invoke(
    "Find the guide in the cold_brew category about how to prepare it."
)

print("\nSelf-query results for cold brew:")
for document in self_query_docs:
    print(
        f"category={document.metadata.get('category', '')} "
        f"source={document.metadata.get('source', '')}"
    )
    print(document.page_content)

