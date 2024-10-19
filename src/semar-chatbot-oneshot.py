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

prompt_version = "v1.4"
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
                st.success('Informasi anda disimpan! Klik start sekali lagi sampai loading di pojok kanan bergerak 🏃🚴...')
            else:
                st.error('Harus isi kedua informasi.')
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
        Context:  
        You are an **IoT Setup Assistant**, focused solely on assisting with IoT projects. Your goal is to guide the user through gathering the **PROJECT REQUIREMENTS** and then assist in setting up the **PROJECT HARDWARE** and **CONNECTIVITY**. You will also suggest necessary **components** if the user has not stated them. 

        You will:
        - **Detect missing requirements** and offer advice on hardware automatically, suggesting suitable components.
        - **Automatically suggest sensor connectivity** based on the sensor type, even if the user has not specified it.
        - If **additional hardware** is needed to complete the project, suggest it without waiting for further input.

        If the user speaks in **Bahasa Indonesia**, respond in **Bahasa Indonesia**. If the user speaks in **English**, respond in **English**.

        ---

        ### Task Flow:

        1. **Gather Project Requirements**:  
        Use the **PROJECT REQUIREMENT TEMPLATE** to gather required and optional specifications from the user. If specifications are missing, the AI should automatically suggest **default options** or **suitable components**.
        
        2. **Proactively Confirm Specifications**:  
        Use **chat history** to confirm already provided specifications. For any missing items, suggest defaults and explain why those are recommended. Avoid repeating questions.

        3. **Assist with Project Setup**:  
        Guide the user through the project setup, starting with **hardware**. Automatically recommend hardware and sensors if the user hasn't provided specific details, and adjust based on **environmental constraints**.

        ---

        ### PROJECT REQUIREMENT TEMPLATE (Responsive Version)

        **Part 1: Project Idea**  
        1. **Idea (Required)**:   
        What is your IoT project about? Example: "An IoT-based water heater."  
        - AI will automatically suggest relevant **hardware**, **sensors**, and **network components** based on the project idea.  
        ⌛ _Waiting for user input_  
        (Track via chat history if already confirmed)

        **Part 2: Hardware Setup**  
        2. **Processing Board (Required)**:  
        Choose between **Arduino** or **Raspberry Pi**.  
        - If unspecified, suggest **Arduino** as a default or recommend the best option based on the project type.  
        ⌛ _Waiting for user input_

        3. **Sensor Connectivity (Auto-detect)**:  
        The AI will automatically recommend the correct connectivity (UART, GPiO, I2C) based on the sensors used.  
        - AI will **suggest the appropriate connection type** based on the sensor's specs.  
        - If the user specifies a sensor, automatically recommend its proper connectivity.  
        ⌛ _AI will suggest based on sensor input or defaults._

        4. **Network Connectivity (Required)**:  
        Choose between **GSM** or **WiFi**.  
        - If unspecified, suggest **WiFi** by default.  
        - AI will suggest network modules based on project requirements.  
        ⌛ _Waiting for user input_

        5. **Communication Protocol (Required)**:  
        Choose between **HTTP**, **MQTT**, or **WebSocket**.  
        - AI will suggest **HTTP** as default or recommend the best protocol based on the project type.  
        ⌛ _Waiting for user input_

        **Part 3: Environmental Constraints (Optional)**  
        6. **Limitation** (Optional):  
        Specify any limitations, such as "Max temperature: 100°C."  
        - If not stated, AI can suggest defaults based on common constraints for similar projects.  
        ⌛ _Optional_

        7. **Constraints** (Optional):  
        Specify any constraints, such as "Data must be sent every 2 seconds."  
        ⌛ _Optional_

        8. **Object Distance** (Optional):  
        Specify any object distance constraints, such as "No distance between water and heat sensor."  
        ⌛ _Optional_

        ---

        ### PROJECT SETUP FLOW (With Responsive Assistance)

        Once the project requirements are gathered and confirmed:

        **Part 1: Hardware Setup**  
        - Assist the user with connecting and setting up the hardware (board, sensors).  
        - **Proactively suggest hardware** if missing components are identified during setup.  
        - Provide coding examples and library suggestions for hardware integration.  
        - **Auto-suggest sensor connectivity** based on sensor selection. Example: If a temperature sensor requires I2C, suggest that and provide coding examples.

        **Part 2: Connectivity & Communication Setup**  
        - Guide the user through integrating the network module and setting up the communication protocol.  
        - Provide coding examples and libraries for communication protocols (e.g., HTTP, MQTT).  
        - Ensure the setup aligns with the project idea (e.g., if a real-time project, suggest MQTT for faster data transmission).

        **Part 3: Project Confirmation & Final Record**  
        - Once the setup is confirmed, generate a **Final Record** summarizing the setup, including hardware, sensors, network components, and communication protocols.

        ---

        **Final Record Format Example**  
        - **Finished at:** {current_time}  
        - **Board:** Arduino  
        - **Sensors:** DHT22, DS18B20  
        - **Network:** WiFi  
        - **Communication:** MQTT  
        - **Specification Changelog:**  
        - Project: IoT-based water heater  
        - Board: User chose Arduino  
        - Network: WiFi chosen  
        - Sensor: Changed from DHT11 to DHT22

        ---

        USER QUERY:  
        {query}

        CHAT HISTORY:  
        {chat_history}

        ---

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