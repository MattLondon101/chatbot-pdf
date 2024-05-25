from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from prettytable import PrettyTable
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from line_profiler import LineProfiler
# import cProfile


df = pd.DataFrame(columns=['Question', 'Answer'])
ocsv = './output/transcript.csv'
df.to_csv(ocsv)


def main():

    load_dotenv()

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask Your PDF")

    pdf = st.file_uploader("Upload your pdf",type="pdf")

    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # spilit ito chuncks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embedding
        embeddings = HuggingFaceEmbeddings()

        knowledge_base = FAISS.from_texts(chunks,embeddings)

        user_question = st.text_input("Ask Question about your PDF:")
        cnt = 0
        if user_question:
            print(f"Question number {cnt}")
            cnt += 1

            docs = knowledge_base.similarity_search(user_question)

            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5,"max_length":64})

            chain = load_qa_chain(llm,chain_type="stuff")

            response = chain.run(input_documents=docs,question=user_question)

            st.write(response)

            ocsv = './output/transcript.csv'
            df = pd.read_csv(ocsv)
            lofr = len(df.index)
            if lofr == 0:
                df = pd.DataFrame([[user_question, response]], columns=['Question', 'Answer'])
            elif lofr > 0:
                df.loc[lofr] = [user_question, response]

            print(f"df = {df}")
            df.to_csv(ocsv, index=False)

        # st.write(chunks)


if __name__ == '__main__':

    main()
    

    # cProfile.run('main()')
    # lp = LineProfiler()
    # lp_wrapper = lp(main)
    # lp_wrapper()
    # lp.print_stats()
