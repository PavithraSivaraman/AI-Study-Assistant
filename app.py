import streamlit as st
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder


@st.cache_resource
def load_pipeline():
    loader = PyPDFLoader("sample.pdf")  
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    return retriever, reranker, llm

retriever, reranker, llm = load_pipeline()


prompt_template = """You are an AI assistant answering questions strictly from context.

TASK:
1. Check if the context contains relevant information about the question.
2. If YES → return only the definition (1–2 lines).
3. If NO → return exactly: Not found in document

----------------------
EXAMPLES:

Example 1:
Context: Network topology is the arrangement of nodes in a network.
Question: What is network topology?
Answer: Network topology is the arrangement of nodes in a network.

Example 2:
Context: Topology affects performance but no definition is given.
Question: What is topology?
Answer: Not found in document

Example 3:
Context: OSI model stands for Open Systems Interconnection and is a conceptual framework.
Question: What is OSI model?
Answer: OSI model is a conceptual framework that standardizes communication functions.

Example 4:
Context: Resource Sharing - Many organization has a substantial number of computers in operations.
Question: What is resource sharing?
Answer: Resource sharing refers to the sharing of resources among multiple computers in a network.

----------------------

STRICT RULES:
- Use ONLY the given context
- Do NOT guess or infer
- Do NOT give partial answers
- Do NOT add extra text
- Output must be a single clean sentence OR "Not found in document"

----------------------

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def rerank_documents(query, docs):
    pairs = [[query, doc.page_content[:512]] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs]

def clean_output(text):
    text = text.split("\n")[0].strip()
    if not text.endswith(('.', '!', '?')):
        text = re.sub(r'\s+\S*$', '', text)
    return text.strip()


st.title("AI Study Assistant")

query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if query:
        docs = retriever.get_relevant_documents(query)
        unique_docs = list({doc.page_content: doc for doc in docs}.values())

        if not unique_docs:
            st.write("Answer: Not found in document")
        else:
            reranked = rerank_documents(query, unique_docs)
            top_docs = reranked[:2]

            context = "\n".join([d.page_content for d in top_docs])
            final_prompt = PROMPT.format(context=context, question=query)

            response = llm.invoke(final_prompt)
            answer = clean_output(response)

            if not answer:
                answer = "Not found in document"

            score = reranker.predict([[query, top_docs[0].page_content]])[0]

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Confidence")
            st.write(round(score, 2))

            st.subheader("Source")
            st.write(top_docs[0].page_content[:300])