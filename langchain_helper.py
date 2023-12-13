from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import json
import os

load_dotenv()
embeddings = OpenAIEmbeddings()
FAISS_DB_INDEX = 'faiss_index'

def create_db_from_documents() -> FAISS:
    with open('./urls2.json') as file:
      file_contents = file.read()
    parsed_json = json.loads(file_contents)

    loader = WebBaseLoader(parsed_json['urls'])
    loader.requests_per_second = 1
    loader.requests_kwargs = {'verify':False}
    sources = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(sources)

    if os.path.exists(FAISS_DB_INDEX):
        db = FAISS.load_local(FAISS_DB_INDEX, embeddings)
        return db

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_DB_INDEX)

    return db


def get_response_from_query(db, query, openai_api_key="", k=4):
    if not get_response_from_query:
      return

    docs = db.similarity_search_with_score(query, k=k)
    print("docs")
    print(docs)
    docs_page_content = " ".join([d.page_content for d, score in docs])

    llm = OpenAI(openai_api_key=openai_api_key)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about Spain Immigration
        based on the documents.

        Answer the following question: {question}
        By searching the following documents: {docs}

        Only use the factual information from the documents to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed. They should be answered in English
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    return response