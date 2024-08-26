from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
import os


load_dotenv()

# Set the API key
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_response(message, history):
    formatted_history = []
    for user, assistant in history:
        formatted_history.append({"role": "user", "content": user })
        formatted_history.append({"role": "assistant", "content":assistant})

    formatted_history.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(model='gpt-3.5-turbo',
    messages= formatted_history,
    stream=True)

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              yield partial_message

gr.ChatInterface(generate_response,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="You can ask me anything", container=False, scale=7),
    title="OpenAI Chat Bot",
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear").launch()
gr.ChatInterface(generate_response).launch()