# from dotenv import load_dotenv
import os
import sys
# import streamlit as st
import pandas as pd
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# if 'count_value' not in st.session_state:
#     st.session_state.count_value = 0


def main(newfi, hftoken, pdfpath, uquestion):

    # load_dotenv()

    # st.set_page_config(page_title="Ask your PDF")
    # st.header("Ask Your PDF")

    # pdf = st.file_uploader("Upload your pdf",type="pdf")

    print(f"sys.argv = {sys.argv}")

    ocsv = './output/transcript.csv'
    if newfi == True:
        if os.path.isfile(ocsv) == False:
            df = pd.DataFrame(columns=['User_Question', 'Chatbot_Answer'])
        elif os.path.isfile(ocsv) == True:
            # if st.session_state.count_value == 0:
            os.remove(ocsv)
            df = pd.DataFrame(columns=['User_Question', 'Chatbot_Answer'])
            # elif st.session_state.count_value > 0:
            #     df = pd.read_csv(ocsv)
    elif newfi == False:
        if os.path.isfile(ocsv) == False:
            df = pd.DataFrame(columns=['User_Question', 'Chatbot_Answer'])



    if pdfpath is not None:

        pdf_reader = PdfReader(pdfpath)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # spilit into chuncks
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

        # user_question = st.text_input("Ask Question about your PDF:")

        if user_question:

            docs = knowledge_base.similarity_search(user_question)

            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5,"max_length":64})

            chain = load_qa_chain(llm,chain_type="stuff")

            response = chain.run(input_documents=docs,question=user_question)

            # st.write(response)

            lofr = len(df.index)
            if lofr == 0:
                df = pd.DataFrame([[user_question, response]], columns=['User_Question', 'Chatbot_Answer'])
            elif lofr > 0:
                df.loc[lofr] = [user_question, response]

            df.to_csv(ocsv, index=False)


if __name__ == '__main__':

    # User input commands
    print(f"\nNOTE:\nYou will need your HUGGINGFACEHUB_API_TOKEN to run this app.\nIf you have a Hugging Face account, it can be found at https://huggingface.co/settings/tokens.\nIf you do not have a Hugging Face account, you can create a free one at https://huggingface.co/.\n")        
    newfi = str(input("Create new transcript.csv? (True or False):"))
    hftoken = str(input("Enter your HUGGINGFACEHUB_API_TOKEN:"))
    pdfpath = str(input("Enter path to PDF:"))
    uquestion = str(input("Enter your question:"))

    main(newfi, hftoken, pdfpath, uquestion)
    
    # st.session_state.count_value += 1
