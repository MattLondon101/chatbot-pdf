# chatbot-pdf

Chat with a PDF and generate a transcript of the conversation. This app can be run locally or with Streamlit in browser.


## Installation

To run this project, please follow the steps below:

1. Clone the repository:

    ```shell
    git clone git@github.com:MattLondon101/chatbot-pdf.git
    cd chatbot-pdf
    ```

2. Create and activate a conda virtual environment (optional but recommended):

    ```shell
    conda create -n env1 python=3.10
    conda activate env1
    ```

3. Install the dependencies from the `requirements.txt` file:  

    NOTE: If you are not using a CUDA supported GPU, in `requirements.txt` line 1, change `faiss-gpu` to `faiss-cpu`.  

    ```shell
    pip install -r requirements.txt
    ```

4. You will need a HUGGINGFACEHUB_API_TOKEN for this next step. To obtain one for free, got to https://huggingface.co/ and Sign Up for a free account. Then, go to Settings > Access Tokens. Create a New token. Then, create a file in this directory, name is `.env` and enter `HUGGINGFACEHUB_API_TOKEN=token`,  replacing `token` with your User Access Token. Save the `.env` file. The `.gitignore` file will ignore the `.env` for git operation.

## Running the Project

Once you have installed the required dependencies, you can run the project using Streamlit, which should have been installed with `requirements.txt`. Streamlit provides an easy way to create interactive web applications in Python.

To start the application, run the following command:

```shell
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser..  


## License

This project is licensed under the [MIT License](LICENSE).

