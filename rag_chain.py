from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model="llama3.2:1b")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a weather research assistant. Use the following context to answer the question. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
)

def ask(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = prompt_template.format(context=context, question=question)
    return llm.invoke(prompt)

if __name__ == "__main__":
    question = "What model performed best for predicting precipitation, and why?"
    answer = ask(question)
    print(f"Q: {question}\n\nA: {answer}")