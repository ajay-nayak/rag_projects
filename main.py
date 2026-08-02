
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

# retrival imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama.chat_models import ChatOllama

# for LCEL function
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


load_dotenv()

prompt_template = ChatPromptTemplate.from_template("""
Given the following query and context, provide a concise answer only based on the given context.
Context:{context}
Query: {query}

Provide a detailed answer:
""")
llm = ChatOllama(model="qwen3:1.7b", temperature=0)
embedding = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vector_store = PineconeVectorStore(embedding=embedding, index_name=os.getenv("INDEX_NAME"))
retriever = vector_store.as_retriever(search_kwargs={"k": 3})    

def ingestion():
    print("Hello from rag-projects!")
    print("Pinecone API Key:", os.getenv("PINECONE_API_KEY"))
    loader = TextLoader("ai_article", encoding="utf-8")
    documents = loader.load()
    print("Documents loaded:", len(documents), "First document content:", documents[0].page_content[:200], "...")
    textSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = textSplitter.split_documents(documents)
    print("Total chunks created:", len(chunks))

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    # embedded_chunks = embeddings.embed_documents(chunks)
    # print("Total embedded chunks created:", len(embedded_chunks))

    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("Pinecone index created and chunks added.")

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        formatted_docs.append(f"Document content: {doc.page_content}\n")
    return "\n\n".join(formatted_docs)

def retrieval_without_lcel(query:str) -> str:
    print("Retrieval function called. This is where you would implement retrieval logic.")
    relevant_docs = retriever.invoke(query)    
    print("Relevant documents retrieved:", len(relevant_docs))



    messages = prompt_template.format(query=query, context=format_docs(relevant_docs))
    response = llm.invoke([HumanMessage(content=messages)])
    print("LLM response:", response)


def retrieval_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    print("Retrieval with LCEL function called. This is where you would implement retrieval logic with LCEL.")
    # Implement LCEL logic here, using the query to retrieve relevant documents and then generate a response based on those documents.

    # This line assigns a new Runnable to the 'context' key using a chain of operations:
# 1. itemgetter("query"): Creates a callable that extracts the "query" key from the input dictionary.
# 2. retriever: Likely a function or Runnable that processes the extracted query (e.g., retrieves relevant data).
# 3. format_docs: Another function or Runnable that formats the data returned by retriever.
# The '|' operator chains these steps, so the output of one is passed as input to the next.
# The result is a Runnable that, when given an input dict, extracts the "query", processes it, and formats the result.
# This entire chain is assigned to the 'context' key in the assign() method, so when the resulting Runnable is called,
# it will add a 'context' field to the output dict, containing the final result of this chain.
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("query") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain



if __name__ == "__main__":
    # ingestion()
    # retrieval_without_lcel("Why AI Requires a New Leadership Model?")
    chain_with_lcel = retrieval_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"query": "Why AI Requires a New Leadership Model?"})
    print("\nAnswer:")
    print(result_with_lcel)
