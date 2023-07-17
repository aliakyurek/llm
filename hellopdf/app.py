import os.path
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import pickle
from dotenv import load_dotenv


@st.cache_data
def get_vector_store(store_name, chunks):
    if os.path.exists(f"{store_name}"):
        with open(f"{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        with open(f"{store_name}.pkl", "wb") as f:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            pickle.dump(vector_store, f)

    return vector_store


def main():
    load_dotenv()

    with st.sidebar:
        st.title("🤗💬 LLM Chat App")
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
        ''')
        add_vertical_space(5)
        st.write('Made with ❤')

    st.header("Hello PDF 💭")

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        vector_store = get_vector_store(pdf.name[:-4], chunks)

        query = st.text_input("Ask questions about your PDF file:")
        if query:
            docs = vector_store.similarity_search(query=query, k=3)

            llm = ChatOpenAI(temperature=0.0)

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            st.write(response)


if __name__ == '__main__':
    main()
