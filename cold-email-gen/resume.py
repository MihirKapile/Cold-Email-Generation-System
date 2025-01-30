import os
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()


class Resume:
    def __init__(self, file_path="resource/resume_mihir.pdf"):
        self.file_path = file_path
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore_path = "vectorstore/faiss_index"

    def load_resume(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=300, chunk_overlap=30, separator="\n"
        )
        docs = text_splitter.split_documents(documents=documents)
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        vectorstore.save_local("vectorstore/faiss_index")

    def query_resume(self, skills):
        new_vectorstore=FAISS.load_local("vectorstore/faiss_index",self.embeddings, allow_dangerous_deserialization=True)
    
        retrieval_qa_chat_prompt= hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain= create_stuff_documents_chain(
            ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-70b-versatile",
        ),retrieval_qa_chat_prompt
        )

        retrieval_chain = create_retrieval_chain(
            new_vectorstore.as_retriever(),combine_docs_chain
        )
        Query = f"""
                Give all the experience , projects and skills from the given resume which is relevant to these given skills {skills}
                """
        res=retrieval_chain.invoke({"input":Query})
        return res['answer']
    
if __name__ == "__main__":
    resume = Resume()
    #resume.load_resume()
    results = resume.query_resume("Python, Machine Learning, Django") 

    print(results)
