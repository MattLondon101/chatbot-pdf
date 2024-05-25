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

# from line_profiler import LineProfiler
# import cProfile


def main():

    load_dotenv()

    df = pd.DataFrame(columns=['Question', 'Answer'])

    # tabl = PrettyTable()
    # tabl.field_names= ["Question", "Answer"]
    # output_file = './output/transcript.txt'
    ocsv = './output/transcript.csv'
    df.to_csv(ocsv)
    # with open(output_file, 'w') as outfi:
    #     outfi.write(str(tabl))

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
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5,"max_length":64})
            chain = load_qa_chain(llm,chain_type="stuff")
            response = chain.run(input_documents=docs,question=user_question)

            st.write(response)

            # ndf = pd.DataFrame(data=[[user_question, response]], columns=None)
            oldFrame = pd.read_csv(ocsv)
            # df_diff = pd.concat([oldFrame, ndf], ignore_index=True)
            # df_diff.to_csv(ocsv)
            
            oldFrame.loc[-1] = [user_question, response]
            # df.t
            print(f"df = {oldFrame}")   

            # table_rows = lambda tabl: len(tabl.get_string().split('\n'))-4
            # table_txt = tabl.get_string(start=table_rows(tabl)-1)
            # print(f"table_txt = {table_txt}")
            # tabl.add_row([user_question, response])

        # st.write(chunks)


if __name__ == '__main__':
    main()

    # cProfile.run('main()')
    # lp = LineProfiler()
    # lp_wrapper = lp(main)
    # lp_wrapper()
    # lp.print_stats()
