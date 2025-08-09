# imports
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import gradio as gr
from dotenv import load_dotenv
import uuid
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import LangChainTracer


# Load environment variables
load_dotenv(override=True)

# Initialize chat history store
store = {}
loader = PyPDFLoader('historical_figures.pdf')
documents = loader.load()
tracer = LangChainTracer(project_name="HistoryBot")

# Split the documents into manageable chunks
# This is important for efficient retrieval and processing
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separator=''
)
splitted_documents = splitter.split_documents(documents)

# Function to get session history
# This function retrieves the chat history for a given session ID
# If the session ID does not exist, it initializes a new InMemoryChatMessageHistory
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Initialize the vector store using Chroma
# This will store the embeddings of the document chunks for efficient retrieval
vector_store = Chroma(
    embedding_function=OllamaEmbeddings(model="granite-embedding:latest"),
    persist_directory='mounika_chroma_db',
    collection_name='historical_figures'
)

vector_store.add_documents(splitted_documents)

# Function to handle user input and generate a response
def chatbot(user_input, history_state, session_id):
    # Check for empty or whitespace-only input
    if not user_input or user_input.strip() == "":
        error_message = "Please enter a valid question!"
        if history_state is None:
            history_state = []
        history_state.append((user_input, error_message))
        return error_message, history_state, session_id
    
    # Initialize the LLM 
    llm = ChatOllama(
        model='llama3.2:latest')
    
    output_parser = StrOutputParser()
    
    # # Create a simple chat prompt
    prompt = PromptTemplate.from_template(
        "You are a HistoryBot an expert on historical figures\n"
        "Context: {context}\n\nQuestion: {question}"
    )
    # Create a retrieval-based question-answering chain
    chain = RetrievalQA.from_chain_type(
        llm=llm | output_parser,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        callbacks=[tracer]
    )

    # Wrap with RunnableWithMessageHistory for session-based chat
    chain = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    # Invoke the chain with user input and session ID
    response = chain.invoke(
        input={"query":user_input},
        config={"configurable": {"session_id": session_id}}
    )
    
    # Extract only the answer from the response
    if isinstance(response, dict) and "result" in response:
        formatted_response = response["result"]
    else:
        formatted_response = str(response)

    if history_state is None:
        history_state = []
    history_state.append((user_input, formatted_response))

    return formatted_response, history_state, session_id

# Function to clear conversation history and input box
def clear_history(session_id):
    # Clear the session history from the store
    if session_id in store:
        store[session_id].clear()
    # Reset history_state, session_id, input_box, and output_box
    return None, str(uuid.uuid4()), "", ""

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## History Chatbot")
    gr.Markdown("### Hello, I am HistoryBot, your expert on historical figures. How can I assist you today?")
    history_state = gr.State(value=None)
    session_id = gr.State(value=str(uuid.uuid4()))
    input_box = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    output_box = gr.Textbox(label="Answer", interactive=False)
    with gr.Row():
        submit_button = gr.Button("Submit")
        clear_button = gr.Button("Clear History")

    submit_button.click(
        fn=chatbot,
        inputs=[input_box, history_state, session_id],
        outputs=[output_box, history_state, session_id]
    )
    clear_button.click(
        fn=clear_history,
        inputs=[session_id],
        outputs=[history_state, session_id, input_box, output_box]
    )

# Launch the Gradio app
demo.launch()