import os
import streamlit as st
import re

from streamlit_chat import message
from pinecone import Pinecone as PineconeClient
from pinecone_text.sparse import BM25Encoder
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
import nltk
nltk.download('punkt_tab')

bm25_encoder=BM25Encoder()
def download_embeddings():
    embedding_path = "local_embeddings"

    # Check if embeddings are already saved locally
    if os.path.exists(embedding_path):
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
    else:
        # Initialize embeddings with the correct model name
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding, f)

    return embedding
embedding=download_embeddings()
def find_match(input_text):
    # Ensure the Pinecone index is correctly initialized
    
    pc = PineconeClient(api_key=st.secrets["PINECONE_API_KEY"])
    
    
    # Ensure the Pinecone index is correctly initialized
    index_name = 'new-hybrid-search'
    index = pc.Index(index_name)
    with open('bm25_encoder.pkl', 'rb') as f:
        bm25_encoder = pickle.load(f)
    
    # Initialize Pinecone retriever
    retriever = PineconeHybridSearchRetriever(
        embeddings=embedding,  # Dense embedding model
        sparse_encoder=bm25_encoder,  # Sparse BM25 encoder
        index=index
    )
    
    # Perform hybrid similarity search
    results = retriever.invoke(
        input=input_text
    )
    return results
from langchain_groq import ChatGroq
api_key1=st.secrets["GROQ_API_KEY"]

# Streamlit setup  

st.subheader("HELPDESK CHAT")

# Initialize session state variables
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi there! Feel free to ask me any question about the Attention Is All You Need research paper."]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize the language model
llm=ChatGroq(groq_api_key=api_key1,model_name="llama3-8b-8192",temperature=0.6)

# Initialize conversation memory
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=1, return_messages=True)

# Define prompt templates
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer as a friendly helpdesk agent and use the provided context to build the answer and If the answer is not contained within the text, say 'I'm not sure about that, but I'm here to help with anything else you need!'. Do not say 'According to the provided context' or anything similar. Just give the answer naturally.""")                                                                        
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
link='meera11.jpg'
# Create conversation chain
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Container for chat history
response_container = st.container()
# Container for text box
text_container = st.container()



with text_container:
    user_query =st.chat_input("Enter your query")

    if user_query:
        with st.spinner("typing..."):
            context = find_match(user_query)
            response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{user_query}")
            


        
        # Append the new query and response to the session state  
        st.session_state.requests.append(user_query)
        st.session_state.responses.append(response)
st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p{
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True
)


# Display chat history
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            with st.chat_message('Momos', avatar=link):
                st.write(st.session_state['responses'][i])
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')