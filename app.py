import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith configuration
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = "MultiBot"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Hugging face embeddings
os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

st.title('🤖 Multibot')

# Initialize session state
if 'proceed_clicked' not in st.session_state:
    st.session_state.proceed_clicked = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Data upload section
    st.subheader("📄 Upload Data")
    files = st.file_uploader('Upload PDF files', type='pdf', accept_multiple_files=True)
    url = st.text_input('Enter a webpage link')
    
    if st.button('Process Data'):
        if not files and not url:
            st.warning("Please upload at least one PDF or enter a URL.")
        else:
            with st.spinner("Processing data..."):
                documents = []
                splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                
                if url:
                    loader = WebBaseLoader(web_path=url)
                    documents = loader.load()
                    if 'chroma' not in st.session_state:
                        st.session_state.chroma = Chroma.from_documents(
                            splitter.split_documents(documents), embeddings)
                    else:
                        st.session_state.chroma.add_documents(
                            splitter.split_documents(documents))
                    documents = []
                    
                if files:
                    for file in files:
                        temp_path = './temp.pdf'
                        with open(temp_path, 'wb') as f:
                            f.write(file.getvalue())
                            loader = PyPDFLoader('./temp.pdf')
                            documents = loader.load()

                            if 'chroma' not in st.session_state:
                                st.session_state.chroma = Chroma.from_documents(
                                    splitter.split_documents(documents), embeddings)
                            else:
                                st.session_state.chroma.add_documents(
                                    splitter.split_documents(documents))
                            documents = []
                
                st.session_state.proceed_clicked = True
                st.success("Data processed successfully!")
    
    # Model configuration
    st.subheader("⚙️ Model Settings")
    temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, value=0.7)
    tokens = st.slider('Max Tokens', min_value=100, max_value=1000, value=500)
    groq_api_key = st.text_input('Groq API Key', type='password')
    model = st.selectbox('Model', [
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "command-r-plus",
        "llama2-70b-4096"
    ])

def get_session_history(session_id) -> BaseChatMessageHistory:
    if 'store' not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Main chat interface
if st.session_state.proceed_clicked and groq_api_key:
    # Setup chains
    contextualize_que_sys_prompt = """
    You are a question rephrasing assistant.
    Given the chat history and a follow-up question, rephrase the follow-up question to be a standalone question.
    If the question is already standalone, return it unchanged.
    Always end with a question mark. Don't use your own knowledge to answer the question. You have to see chat history and then reframe the question.
    """
    
    contextualize_que_prompt = ChatPromptTemplate.from_messages([
        ('system', contextualize_que_sys_prompt),
        MessagesPlaceholder('chat_history'),
        ('user', "{input}")
    ])

    llm = ChatGroq(model_name=model, api_key=groq_api_key, temperature=temperature, max_tokens=tokens)
    retriever = st.session_state.chroma.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_que_prompt)

    qa_sys_prompt = """
    You are an expert assistant helping with queries.
    ONLY use the provided context below to answer the user's question.
    If the context does not help answer the question, reply:
    "I cannot answer this question as the context does not contain the relevant information."
    NEVER use general knowledge. Do not guess. Be concise and factual.
    
    Ans format:
        Final question
        \n
        Answer
    
    Context: {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', qa_sys_prompt),
        MessagesPlaceholder('chat_history'),
        ('user', '{input}')
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    bot = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    if 'config' not in st.session_state:
        st.session_state.config = {'configurable': {'session_id': 'user123'}}

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = bot.invoke({'input': prompt}, config=st.session_state.config)
                answer = response['answer']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

elif not st.session_state.proceed_clicked:
    st.info("👈 Please upload your documents and process them using the sidebar to start chatting.")
elif not groq_api_key:
    st.info("👈 Please enter your Groq API key in the sidebar to start chatting.")
else:
    st.info("Please configure the settings in the sidebar to start chatting.")
    
    
    
    
# debugging
 # # ===Debugging===
        # # Step 1: Get chat history
        # chat_history = session_history.messages
        # with col1:
        #     st.write(session_history.messages)
        # # Step 2: Use the contextualization prompt to rephrase the question
        # contextualized_question = llm.invoke(contextualize_que_prompt.format_messages(
        #     input=input,
        #     chat_history=chat_history
        # ))

        # # Extract actual text from model output
        # rephrased_question = contextualized_question.content
        # st.subheader("📝 Rephrased Question (After Contextualization)")
        # st.write(rephrased_question)

        # # Step 3: Retrieve documents using rephrased question
        # retrieved_docs = retriever.invoke(rephrased_question)

        # st.subheader("📚 Retrieved Documents After Contextualization")
        # for i, doc in enumerate(retrieved_docs):
        #     st.markdown(f"**Chunk {i+1} (source: {doc.metadata}):**")
        #     st.markdown(doc.page_content)
    
        # # ===Debugging===