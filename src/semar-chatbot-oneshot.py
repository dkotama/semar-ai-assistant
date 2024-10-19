import os
import streamlit as st
import tiktoken
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from pytz import timezone


# Load environment variables
load_dotenv()

# Load the MongoDB client
MONGODB_URI = os.getenv("MONGODB_URI")

# MongoDB Setup
client = MongoClient(MONGODB_URI)
db = client["semar_bot_db"]
sessions_collection = db["sessions"]

prompt_version = "v1.5"
current_model = "gpt-4o"

# Get Current Time Functions
def get_current_time():
    return datetime.now(tz=timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S")


# Set up the page
st.set_page_config(page_title="Semar-Bot", page_icon=":robot:")
st.title("Semar-Bot")

# Step 1: Student ID and Name Input
if 'student_info' not in st.session_state:
    st.session_state['student_info'] = {}

if not st.session_state['student_info']:
    with st.form('student_form'):
        st.write("Masukkan NIM anda dan Nama anda:")
        student_id = st.text_input('NIM')
        student_name = st.text_input('Nama')
        submitted = st.form_submit_button('Start Chat')

        if submitted:
            if student_id.strip() != "" and student_name.strip() != "":
                st.session_state['student_info'] = {
                    'student_id': student_id.strip(),
                    'student_name': student_name.strip()
                }
                st.success('Terima kasih! Informasi anda disimpan!\nKlik start sekali lagi sampai loading di pojok kanan bergerak 🏃🚴...')
            else:
                st.error('Informasi NIM & Nama harus diisi.')
    st.stop()  # Stop execution until the student info is provided

# Step 2: Create a New Session in MongoDB
if 'session_id' not in st.session_state:
    # Create a new session document in MongoDB
    session_data = {
        'student_id': st.session_state['student_info']['student_id'],
        'created_at': get_current_time(),
        'model': current_model,
        'prompt_version': prompt_version,
        'chat_history': []
    }
    session = sessions_collection.insert_one(session_data)
    st.session_state['session_id'] = str(session.inserted_id)

# Step 3: Initialize Chat History
if 'chat_history' not in st.session_state:
    # Fetch the session from MongoDB
    session = sessions_collection.find_one({'_id': ObjectId(st.session_state['session_id'])})
    if session and 'chat_history' in session and len(session['chat_history']) > 0:
        st.session_state['chat_history'] = session['chat_history']
    else:
        st.session_state['chat_history'] = [
            {
                'role': 'assistant',
                'content': f"""
                        Halo! {st.session_state['student_info']['student_name']}! Saya Semar-Bot, asisten Setup IoT mu. Mari mulai dengan setup proyek IoT Anda.
                        Apa jenis proyek IoT yang sedang Anda kerjakan hari ini? Anda bisa mulai dengan menyatakan ide Anda untuk proyek tersebut.
                        sebagai contoh, "Saya ingin membuat pemanas air berbasis IoT, dengan ESP32 sebagai board mikro, dan DHT22 sebagai sensor suhu."
                    _Loaded prompt version: {prompt_version}_
                    """
            }
        ]
# Step 4: Define the get_response Function
def get_response(query, chat_history):

    template = """
        CONTEXT:
        You are an IoT Setup Assistant. Your sole focus is assisting with IoT projects. Always respond in the language the user uses. If the user speaks in Bahasa Indonesia, respond exclusively in Bahasa Indonesia. If the user speaks in English, respond exclusively in English. This rule is critical to follow.

        You must always be **proactive and helpful**. At any point during the conversation, if the user expresses any issues, such as **hardware setup problems** or **code errors**, you should offer help to troubleshoot or resolve these problems. Always offer assistance **before proceeding to the next step**.

        **It is critical to generate a FINAL RECORD** summarizing the project once all steps are complete. Ensure this step is not skipped, and the FINAL RECORD includes all the project’s confirmed specifications, hardware setup, and any environmental constraints provided by the user.

        ---

        TASK:
        Your task is to assist the user in creating a complete IoT project setup by first gathering **PROJECT REQUIREMENTS** and then proceeding with **PROJECT SETUP**.  
        Use CHAT HISTORY to avoid repeating questions.

        For required specifications, continue prompting the user until the information is provided. Suggest appropriate hardware, sensors, or network components where necessary, and use defaults if the user is unsure. For optional specifications, proceed with defaults unless specified otherwise.

        You must proceed step-by-step, ensuring each part is confirmed before moving to the next. **Do not proceed to hardware setup or connectivity setup until all requirements are gathered.**

        Always offer help with any issues the user may encounter before moving forward, whether in the hardware setup, connectivity, or any other step.

        ---

        ### PROJECT REQUIREMENTS TEMPLATE:

        If a specification is confirmed, display it like this:
        1. **Idea (Confirmed)** ✅  
        (Confirmed from Chat History)

        If still waiting, display it like this:
        1. **Idea (Waiting for Confirmation)** ⌛  
        (Waiting for user input)

        ---

        ### PART 1 - IDEA:

        1. **Idea (Required):**  
        Ask the user to state their IoT project idea. Example:  
        "What is your IoT project idea? For example: 'I want to make an IoT-based water heater.'"

        - Wait for confirmation of the idea before proceeding to hardware setup.

        ---

        ### PART 2 - HARDWARE REQUIREMENTS:

        2. **Hardware Requirements (Required):**

        - **Processing Board (Required, default: Arduino):**  
        Choose between Arduino or Raspberry Pi. Default to Arduino if unspecified. If the user's chosen board is incompatible with the sensor given or connection type, explain the limitation and suggest alternatives (e.g., switching boards or adding modules).

        - **Sensor Connectivity (Required, default: GPiO):**  
        Choose between UART, GPiO, or i2C. Recommend the connection type based on the sensor given. If GPiO is required and the user refuses, explain the limitation and suggest alternatives. Confirm how they wish to proceed.

        - **Network Connectivity (Required, default: Wifi):**  
        Choose between GSM or Wifi. Default to Wifi unless GSM is required by the project.

        - **Communication Protocol (Required, default: HTTP):**  
        Choose between HTTP, MQTT, or Websocket. Default to HTTP unless the project requires real-time communication, where MQTT may be preferred.

        - Wait for the user to confirm hardware requirements before proceeding to environmental constraints.

        ---

        ### PART 3 - ENVIRONMENTAL CONSTRAINTS (Optional):

        3. **Environmental Constraints (Optional):**  
        - **Limitation**: (e.g., Max heat 100°C)  
        - **Constraint**: (e.g., Data must be sent every 2 seconds)  
        - **Object Distance**: (e.g., No distance between water and sensor)  
        (Skip this section if the user does not provide specific information.)

        Proceed to the  PROJECT SETUP part only after the user confirms all the required environmental constraints.

        ---

        ### PROJECT SETUP:

        PART 1 - HARDWARE SETUP:
        Once all requirements are gathered, provide the hardware setup based on the confirmed specifications. 
        **Evaluate the feasibility** of the current hardware based on the environmental constraints provided by the user (e.g., temperature, object distance, data frequency). If the current hardware is not suitable, suggest alternative hardware that meets the project requirements. 

        Include board connections, sensor pin setup, necessary libraries, and sample code only for hardware setup.

        - **Board and Sensor Pin Suggestions:**  
        Suggest GPIO pins based on best practices. Example:  
        "For DHT22, use GPIO4 for data, connect VCC to 3.3V, and GND to GND on the ESP32."

        - **Feasibility Check for Environmental Constraints:**  
        If the environmental constraints provided by the user (such as temperature or distance) exceed the capabilities of the current hardware, alert the user and suggest more suitable hardware. 

        - **Other Setup Steps:**  
        Provide instructions for connecting additional components (e.g., resistors, capacitors). Include sample code where necessary.

        Proceed to the connectivity and communication setup only after the user confirms the hardware setup is success.
        ---

        PART 2 - CONNECTIVITY AND COMMUNICATION SETUP:
        Provide instructions for setting up network connectivity and communication protocols (e.g., Wi-Fi setup, MQTT setup). Offer sample code for this and final coding combination with previous hardware coding.

        ---

        USER QUERY:  
        {query}

        CHAT HISTORY:  
        {chat_history}

        """
    # Get the current system time
    current_time =  get_current_time()

    # Create the prompt
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name=current_model)  # Adjust the model name as needed

    chain = prompt | llm | StrOutputParser()

    # Convert chat history to the format expected by the chain
    formatted_chat_history = []
    for msg in chat_history:
        if msg['role'] == 'user':
            formatted_chat_history.append(HumanMessage(content=msg['content']))
        else:
            formatted_chat_history.append(AIMessage(content=msg['content']))

    # Generate the response
    response = chain.invoke({
        "chat_history": formatted_chat_history,
        "query": query,
        "current_time": current_time  # Pass the current time to the prompt
    })

     # Token Counting
    encoding = tiktoken.encoding_for_model(current_model)  # Use the appropriate model name

    # Prepare the full prompt text
    # Include the formatted chat history and the query
    full_prompt_messages = []

    # Add the system prompt (if any)
    full_prompt_messages.append(template)

    # Add the chat history messages
    for msg in formatted_chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        full_prompt_messages.append(f"{role}: {msg.content}")

    # Add the user's query
    full_prompt_messages.append(f"User: {query}")

    # Combine all messages into a single string
    full_prompt_text = "\n".join(full_prompt_messages)

    # Count tokens in the input (prompt + chat history + query)
    input_token_count = len(encoding.encode(full_prompt_text))

    # Count tokens in the assistant's response
    output_token_count = len(encoding.encode(response))

    # Return response and token counts
    return response, input_token_count, output_token_count

# Step 5: Display the Conversation
# Conversation
for message in st.session_state['chat_history']:
    if message['role'] == 'user':
        with st.chat_message("Human"):
            st.markdown(message['content'])
    else:
        with st.chat_message("AI"):
            st.markdown(message['content'])

# Step 6: Handle User Input and Save to MongoDB
# User input
# User input
user_query = st.chat_input("Pesan Anda:")

if user_query is not None and user_query.strip() != "":
    # Token count for user message
    encoding = tiktoken.encoding_for_model(current_model)
    user_token_count = len(encoding.encode(user_query))

    # Append user message to chat history
    user_message = {
        'role': 'user',
        'content': user_query,
        'token_count': user_token_count
    }
    st.session_state['chat_history'].append(user_message)

    with st.chat_message("Human"):
        st.markdown(user_query)

    # Get AI response and token counts
    ai_response, input_token_count, output_token_count = get_response(user_query, st.session_state['chat_history'])

    # Token count for AI message
    ai_token_count = output_token_count

    # Append AI message to chat history
    ai_message = {
        'role': 'assistant',
        'content': ai_response,
        'token_count': ai_token_count
    }
    st.session_state['chat_history'].append(ai_message)

    with st.chat_message("AI"):
        st.markdown(ai_response)

    # Update chat history and token counts in MongoDB
    sessions_collection.update_one(
        {'_id': ObjectId(st.session_state['session_id'])},
        {
            '$set': {'chat_history': st.session_state['chat_history']},
            '$inc': {
                'total_input_tokens': input_token_count,
                'total_output_tokens': output_token_count
            },
            '$push': {
                'token_usage': {
                    'timestamp': get_current_time(),
                    'input_tokens': input_token_count,
                    'output_tokens': output_token_count,
                    'total_tokens': input_token_count + output_token_count
                }
            }
        }
    )