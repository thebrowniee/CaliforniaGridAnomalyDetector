import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="/Users/ruchi/Desktop/chroma_db",
    embedding_function=embeddings
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt_template = """You are an energy grid analyst. Use the context below to explain the anomaly described in the question. Be specific and draw directly from the context provided. If the context discusses heat, demand, or grid conditions, use that to build your explanation.

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}

)

anomalies_to_explain = [
    "What causes overnight electricity demand to spike unusually high in California during summer? What weather or grid conditions lead to high demand at 2am?",
    "Why would California midday electricity demand drop significantly below average in summer? What factors cause unusually low demand at 1pm?",
]

for question in anomalies_to_explain:
    print(f"\nQuestion: {question}")
    print("-" * 60)
    result = qa_chain.invoke({"query": question})
    print(f"Answer: {result['result']}")
    print(f"Sources used: {len(result['source_documents'])} chunks")
    print("\nActual chunks used:")
for doc in result['source_documents']:
    print(f"\n  Source: {doc.metadata['source']}")
    print(f"  Text: {doc.page_content[:200]}")