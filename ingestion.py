import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (
    Colors,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)

load_dotenv()


# configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.getenv("SSL_CERT_FILE", certifi.where())
os.getenv("REQUESTS_CA_BUNDLE", certifi.where())


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    chunk_size=50,
    retry_min_seconds=10,
    retry_max_seconds=60,
)

vector_store = Chroma(
    persist_directory="chroma_bd", embedding_function=embeddings
)

# vector_store = PineconeVectorStore(
#     index_name=os.getenv("PINECONE_DB_INDEX_NAME"),
#     embedding=embeddings,
#     namespace="rag-web-search-helper",
# )

# tavily_extract = TavilyExtract()
# tavily_map = TavilyMap(
#     max_depth=2,
#     max_breadth=5,
#     max_pages=10,
# )
tavily_crawl = TavilyCrawl()


async def index_documents_async(
    documents: List[Document], batch_size: int = 50
) -> None:
    """Asynchronously index documents into the vector store."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"Vector store: Preparing to add {len(documents)} documents in batches of {batch_size}...",
        Colors.DARKCYAN,
    )
    # Create batches
    batches = [
        documents[i : i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"Vector store: Split into {len(batches)} batches of {batch_size} documents each."
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int) -> bool:
        try:
            # TODO: the Pinecone Vector Store session will be closed in one of concurrent tasks.
            # The PineconeVectorStore client (and underlying Pinecone SDK session) is not concurrency-safe
            # — it uses an internal HTTP session that is closed when any task finishe.
            # FIX: initial each vector store instance in each task
            # or most reliable fix is to index batches sequentially instead of concurrently.
            await vector_store.aadd_documents(batch)
            log_success(
                f"Batch {batch_num + 1}/{len(batches)} indexed successfully."
            )
            return True
        except Exception as e:
            log_error(f"Batch {batch_num + 1} failed to index: {e}")
            return False

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"Vector store Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"Vector store Indexing: Processed {successful}/{len(batches)} batches successfully"
        )

    # loop = asyncio.get_event_loop()
    # await loop.run_in_executor(None, vector_store.add_documents, documents)
    # log_success(f"Indexed {len(documents)} documents into the vector store.")


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🔍 TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )
    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://python.langchain.com/en/latest/index.html",
            "max_depth": 2,
            "extract_depth": "basic",
        }
    )

    all_docs = []
    for tavily_crawl_result_item in tavily_crawl_results["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        if (
            not tavily_crawl_result_item["raw_content"]
            or not tavily_crawl_result_item["url"]
        ):
            continue
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )
    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    txt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50
    )
    splitted_docs = txt_splitter.split_documents(all_docs)

    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {len(tavily_crawl_results)}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
