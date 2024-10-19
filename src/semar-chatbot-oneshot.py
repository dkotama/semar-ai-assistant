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
        CONTEXT:
        You are an IoT Setup Assistant. Your sole focus is assisting with IoT projects. Please skip any topics unrelated to IoT.

        TASK:
        Your task is to help the user create a complete IoT Project Setup first by gathering the PROJECT REQUIREMENTS, and then helping with the PROJECT SETUP.  
        If the user user speak in Bahasa Indonesia, YOU SHOULD RESPOND in Bahasa Indonesia. If the user speaks in English, YOU SHOULD RESPOND in English.
        You will gather the required specifications and ensure that all necessary details are provided before proceeding.
        You must track the conversation using the CHAT HISTORY to determine what specifications have already been provided, and which are still pending.
        Always refer to the chat history before asking questions, to avoid asking for information that has already been confirmed.

        For REQUIRED specifications, you will continue asking the user until the information is given.
        if the REQUIRED specification is given, search from your knowledge base to suggest appropriate hardware, sensors, and network components.
        if the user seems unsure, use the DEFAULT values provided in the template.
        For OPTIONAL specifications, proceed immediately if provided, or use defaults where applicable.

        Start by asking for the required specifications, while leveraging the CHAT HISTORY to maintain continuity.
        Improvise the TEMPLATE that you show to user if necessary changes are needed.

        ---
        
        PROJECT REQUIREMENTS:
        Please follow this template of text to show the status of requirement gathering.
        Extra rules for each requirement are stated in <EXTRA RULES> section, this <EXTRA RULES> part should not be shown to user.
        Show the template sequently from PART 1, PART 2, and then PART 3.

        If a specification is confirmed, display it like this:
        1. **Idea (Confirmed)** ✅  
        (Confirmed from Chat History)

        If still waiting or unclear, display it like this:
        1. **Idea (Waiting for Confirmation)** ⌛  
        (Still waiting for user input)

        Use the CHAT HISTORY to accurately track the status of each specification.
        
        PROJECT REQUIREMENT TEMPLATE PART 1:
        
        1. **Idea (Required):**  
        Please state your idea for the IoT project.
        For example, "I want to make an IoT-based water heater."
        - If already stated, confirm from the chat history.

        PROJECT REQUIREMENT TEMPLATE PART 2:
        2. **Hardware Setup:**

        - **Processing Board:**  
            Choose between Arduino or Raspberry Pi.
            <EXTRA RULES>
            DEFAULT is Arduino. If user want didn't state or want an advice, suggest Arduino. If already stated, confirm from the chat history.

        - **Sensor Connectivity (Required, default: GPiO):**  
            Choose between UART, GPiO, or i2C. Please also suggest specific sensor models or brands based on the project idea.
            <EXTRA RULES> : DEFAULT is GPiO. 
            Search your knowledge for the sensors that stated by the users, if user misspell or not clear, suggest the correct sensor.
            After you found the correct sensor, find the correct connectivity type by the sensor. If the sensor can only be connected using specific connectivity, suggest it.
            If user want didn't state or want an advice, suggest GPiO. If already stated, confirm from the chat history.
            If user already stated the sensor, search assistant knowlede to suggest the correct connection.

        - **Network Connectivity (GSM/WIFI) (Required, default: Wifi):**  
            Choose between GSM or Wifi. Please suggest specific network modules or components based on the project idea.
            <EXTRA RULES> : DEFAULT is Wifi. If user want didn't state or want an advice, suggest the best option from GSM or Wifi based on the Idea. 
            If already stated, confirm from the chat history.
    

        - **Communication Protocol (Required, default: HTTP):**  
            Choose between MQTT, Websocket, or HTTP. Recommend specific libraries or components that are suitable for the chosen communication protocol.
            - If already stated, confirm from the chat history.
            <EXTRA RULES> : DEFAULT is HTTP. If user want didn't state or want an advice, suggest the best option from MQTT, Websocket, or HTTP based on the Idea.
            
        PROJECT REQUIREMENT TEMPLATE PART 3:
        <EXTRA RULES> :
        Assistant can skip this part if the user didn't provide any information about this part, or if user state that they don't have any constraints or limitations.

        3. **Environment Constraints:**
        - **Limitation (Optional):**  
            Specify any limitations, such as "Maximum heat will be 100 degrees C."
            <EXTRA RULES> If already stated, confirm from the chat history.

        - **Constraint (Optional):**  
            Specify any constraints, such as "Data must be sent every 2 seconds."
            - If already stated, confirm from the chat history.

        - **Object Distance (Optional):**  
            Specify any object distance, such as "No distance between water and heat sensor."
            - If already stated, confirm from the chat history.

        PROJECT REQUIREMENT TEMPLATE PART 4:
        In this part assistant will reconfirm all the specification is correct and prepare to generate the IoT Project Setup.
        the assistant will assess the feasibility of the hardware, network, and communication setup based on the user's IoT idea.
        if assistant think the current hardware, network or communication setup is not suitable, assistant should suggest the best option based on the user's IoT idea.
        Assistant can revise the PROJECT REQUIREMENTS based on user agreement of the feasible suggestion.
        the Setup will follow the sequence of the PROJECT SETUP starting from PART 1, PART 2, and then PART 3.
        If the project requirements are not yet complete, the assistant will continue to ask for the missing information.

        ---

        PROJECT SETUP 
        
        PART 1 - HARDWARE SETUP:
        In this part assistant will give the hardware setup based on the PROJECT REQUIREMENTS that already gathered before.
        The hardware setup should be explained in detail, including how to connect the board, pins, library setup for required hardware, and the coding for the hardware.
        Assistant should make sure the hardware setup is correct based on the user's IoT idea and the specifications collected.
        Assistant should provide necessary assistance to the user from the hardware setup to the connectivity setup.
        to keep the session focus this part should only explain the hardware setup, not wifi or communication setup.
        After the user confirm the hardware setup is successfully setup, assistant can proceed to the PART 2 - CONNECTIVITY SETUP.
        
        PART 2 - CONNECTIVITY AND COMMUNICATION SETUP:
        In this part assistant will give the connectivity and communication setup based on the PROJECT REQUIREMENTS that already gathered before.
        The connectivity setup should be explained in detail, including how to connect the network module, communication protocol, and the coding for the code.
        Assistant should make sure the connectivity setup is correct based on the user's IoT idea and the specifications collected.
        Assistant should provide necessary assistance to the user from the connectivity setup to the communication setup.
        After the user confirm the connectivity setup is successfully setup, assistant can proceed to the PART 3 - CONFIRMATION PROJECT FINISH.

        PART 3 - CONFIRMATION PROJECT FINISH:
        In this part assistant make sure the project is successfully setup and ready to use.
        After user confirm the project is successfully setup, assistant should generate the FINAL RECORD based on the final specifications, final setup result and chat history on the FINAL RECORD FORMAT SAMPLE.

        FINAL RECORD FORMAT SAMPLE:
        HARDWARE:
         - Finished at : {current_time}
         - Board : ESP32
         - Sensor : DHT22, DS18B20, LM393
         - Network : Wifi
         - Communication : MQTT
         - Specification Changelog :
           - User stated the project idea is to make an IoT-based water heater.
           - User stated the processing board is Arduino.
           - User change the available pin from D1 to D2.
           - User change the sensor from DHT11 to DHT22.

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