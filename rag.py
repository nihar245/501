from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
import os

curdir = os.path.dirname(__file__)

filepath = os.path.join(curdir,"external","a.txt")
pers_dir = os.path.join(curdir,"db4","chroma_db1")

if not os.path.exists(pers_dir):
    print("Vector store does not exist")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"The file {filepath} does not exists."
        )
    loader = TextLoader(file_path=filepath)
    doc=loader.load()

    print("Document length:",len(doc))
    print("Content:",doc[0].page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=0)
    chunks = text_splitter.split_documents(doc)

    print("chunks count:", len(chunks))
    if len(chunks) > 0:
        print("first chunk snippet:")
        print(chunks[0].page_content[:1000])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=pers_dir)
    print("Vector store complete.")

else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=pers_dir, embedding_function=embeddings)

query = "How was Ayodhya?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.3},
)

print(type(retriever))

retrieved_docs = retriever.invoke(query)

print("Retrieved Documents:")
for i,rdoc in enumerate(retrieved_docs,1):
    print ("Document:",i)
    print(rdoc.page_content)
    print("Source:",rdoc.metadata['source'])

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama-3.3-70b-versatile",api_key=groq_api_key)

context = " ".join([rdoc.page_content for rdoc in retrieved_docs])

response = model.invoke([
    {"role":"system","content":"You are a helpful assistant. Use the provided context to answer the user's question accurately."},
    {"role":"user","content":f"content\n{context}\n\nQuestion:\n{query}"}
])

print(response.content if hasattr(response,"content") else response)