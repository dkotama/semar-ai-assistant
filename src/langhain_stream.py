import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Langchain Streamlit", page_icon=":robot:")



st.title("Langchain Streamlit") 

# get response
def get_response(query, chat_history):
    template = """
        You are a helpful Chef Assistant.
        You will guide the user step-by-step through the process of cooking their dish, starting with the ingredients and proceeding with clear instructions.
        
        User Question:
        {query}

        Use the following chat history as context for your response:
        Chat History:
        {chat_history}

        Be sure to provide any missing details or instructions and build on the previous steps if applicable.
        Always prioritize accuracy and clarity, and ensure that the user understands the steps and ingredients.

        Your response should include:
        - The ingredients if the user asks for them.
        - The next step if the user is mid-way through the recipe.
        - A friendly tone that encourages the user.
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-4o")

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "chat_history": chat_history, 
        "query": query
    })

    return chain.invoke({"chat_history": chat_history, "query": query})

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













