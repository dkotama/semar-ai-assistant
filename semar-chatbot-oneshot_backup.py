import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

prompt_version = "v1.4"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(f"""
            Halo ! Saya adalah SEMAR-Bot! asisten setup IoT Anda. \n
                Proyek apa yang ingin Anda bangun hari ini?.\nAnda bisa memulainya dengan mengatakan "Saya ingin membuat IoT untuk mengukur suhu saya"
            \n\n\n_Loaded prompt version: {prompt_version}_
            """)
    ]

st.set_page_config(page_title="Semar-Bot", page_icon=":robot:")



st.title("Langchain Streamlit") 

# get response
def get_response(query, chat_history):
    template = """
            You are an IoT Setup Assistant. Your sole focus is assisting with IoT projects. Please skip any topics unrelated to IoT.
            If the user user speak in Bahasa Indonesia, please respond in Bahasa Indonesia. If the user speaks in English, please respond in English.

            TASK:
            Your task is to help the user create a complete IoT Project Document by conducting an interview.  
            You will gather the required specifications and ensure that all necessary details are provided before proceeding.
            You must track the conversation using the **Chat History** to determine what specifications have already been provided, and which are still pending.
            Always refer to the chat history before asking questions, to avoid asking for information that has already been confirmed.

            For REQUIRED specifications, you will continue asking the user until the information is given.
            if the REQUIRED specification is given, search from your knowledge base to suggest appropriate hardware, sensors, and network components.
            For OPTIONAL specifications, proceed immediately if provided, or use defaults where applicable.

            Start by asking for the required specifications, while leveraging the chat history to maintain continuity.
            ---

            User's Query:  
            {query}

            Chat History:  
            {chat_history}

            ---

            ### Interview Steps:

            1. **Idea (Required):**  
            Please state your idea for the IoT project. For example, "I want to make an IoT-based water heater."
            - If already stated, confirm from the chat history.

            2. **Hardware Setup:**

            - **Processing Board (Required, default: Arduino):**  
                Choose between Arduino or Raspberry Pi.
                - If already stated, confirm from the chat history.

            - **Sensor Connectivity (Required, default: GPiO):**  
                Choose between UART, GPiO, or i2C. Please also suggest specific sensor models or brands based on the project idea.
                - If already stated, confirm from the chat history.

            - **Network Connectivity (Required, default: Wifi):**  
                Choose between GSM or Wifi. Please suggest specific network modules or components based on the project idea.
                - If already stated, confirm from the chat history.

            - **Communication Protocol (Required, default: HTTP):**  
                Choose between MQTT, Websocket, or HTTP. Recommend specific libraries or components that are suitable for the chosen communication protocol.
                - If already stated, confirm from the chat history.

            3. **Environment Constraints:**
            - **Limitation (Optional):**  
                Specify any limitations, such as "Maximum heat will be 100 degrees C."
                - If already stated, confirm from the chat history.

            - **Constraint (Optional):**  
                Specify any constraints, such as "Data must be sent every 2 seconds."
                - If already stated, confirm from the chat history.

            - **Object Distance (Optional):**  
                Specify any object distance, such as "No distance between water and heat sensor."
                - If already stated, confirm from the chat history.

            ---

            ### Brand and Model Suggestions:

            Based on the user's IoT idea and the specifications collected, suggest appropriate hardware, sensors, and network components, including specific brands and models.

            For example:
            - **Sensors:**  
            For temperature sensing in a water heater project, consider using the **DS18B20 Waterproof Temperature Sensor** or the **DHT22 Humidity and Temperature Sensor**. Here's how you can integrate these models with the Arduino or Raspberry Pi.

            - **Network Modules:**  
            For WiFi connectivity, consider using the **ESP8266 WiFi Module** or the **ESP32 Development Board**, both of which are compatible with Arduino and Raspberry Pi.

            - **Communication Protocol:**  
            If the user selects MQTT, recommend using the **PubSubClient library** for Arduino or the **paho-mqtt library** for Raspberry Pi.

            ---

            ### Status Display:

            If a specification is confirmed, display it like this:
            1. **Idea (Confirmed)** ✅  
            (Confirmed from Chat History)

            If still waiting or unclear, display it like this:
            1. **Idea (Waiting for Confirmation)** ⌛  
            (Still waiting for user input)

            Use the chat history to accurately track the status of each specification.

            ---

            ### Proceeding to Document Generation:

            Once all required specifications are gathered, confirm with the user if they are ready to generate the IoT Project Document by typing "Proceed". If any required information is missing, continue to ask for it until provided.

            ---

            ### IoT Project Document Format:

            Once all required information is collected, look from your knowledge base how to integrate:
            - Correct sensors (specific brands, types, and models) according to the project idea.
            - Correct processing board (brand and model) according to the project idea.
            - Correct network connectivity components and integration methods.
            - Correct communication protocols and libraries required for implementation.

            The project document will be generated in this format:

            ---

            ### IoT Project Document:

            1. **IoT Idea:** (State the confirmed idea from the user based on the chat history.)
            2. **Hardware Setup:**
            1. **Processing Board:** (State the confirmed processing board. Suggest a specific model such as Arduino Uno or Raspberry Pi 4.)
            2. **Sensor Connectivity:** (State the confirmed sensor connectivity and list specific sensor models, such as DS18B20 or DHT22.)
            3. **Network Connectivity:** (State the confirmed network connectivity and recommend specific modules such as ESP8266 or ESP32.)
            4. **Communication Protocol:** (Describe the chosen communication protocol, such as MQTT, and recommend libraries like PubSubClient or paho-mqtt.)

            3. **Environment Constraints:**
            1. **Limitation:** (State any limitations provided by the user.)
            2. **Constraint:** (State any constraints provided by the user.)
            3. **Object Distance:** (State any object distance provided by the user.)

            ---

            ### Steps to Integrate:

            Provide general steps based on typical IoT projects:
            1. Step 1: Install and configure the processing board (e.g., Arduino Uno or Raspberry Pi 4).
            2. Step 2: Connect the sensors to the processing board based on the chosen connectivity (e.g., GPiO, UART, or i2C). Provide detailed instructions on wiring and integration with the sensors like DS18B20 or DHT22.
            3. Step 3: Set up network connectivity (e.g., using ESP8266 or ESP32) and ensure the board can communicate with the network.
            4. Step 4: Implement the communication protocol (e.g., MQTT via PubSubClient or paho-mqtt) and test data transmission.

            ### Code on the Hardware:

            Provide generic code snippets based on the hardware setup and communication protocol:
            1. Basic code to read sensor data from specific sensors (e.g., DS18B20, DHT22).
            2. Code to establish network connectivity using the selected module (e.g., ESP8266 or ESP32).
            3. Code to implement the chosen communication protocol (e.g., MQTT, WebSocket, or HTTP).
            4. Code to manage timing and data transmission intervals.

            ---

            You should continue refining the project document as you gather more specifications or user input.

            """


    # required_specifications = {
    #     "idea": "IoT project idea",
    #     "processing_board": "processing board choice (Arduino or Raspberry Pi), default Arduino",
    #     "sensor_connectivity": "sensor connectivity (UART, GPiO, i2C), default i2c",
    #     "network_connectivity": "network connectivity (Wifi or GSM), default Wifi",
    #     "communication_protocol": "communication protocol (MQTT, HTTP, or Websocket), default HTTP"
    # }

    # missing_specs =  [spec for spec, desc in required_specifications.items() 
    #                  if desc.lower() not in ' '.join(str(msg.content).lower() for msg in chat_history)]

    # if missing_specs:
    #     missing_specs_string= ", ".join(required_specifications[spec] for spec in missing_specs)
    #     return f"Missing specifications: {missing_specs_string}. Please provide the missing details."

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-4o-mini")

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "chat_history": chat_history, 
        "query": query
    })


# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)




# user input
user_query = st.chat_input("Your message:")

# render user input and ai response
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)


    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history)
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))













