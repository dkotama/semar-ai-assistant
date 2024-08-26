import openai
import time
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

# gets API Key from environment variable OPENAI_API_KEY
client = openai.OpenAI()

assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions. If user ask you question outside of math, you should ask the user to ask a math question.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
)

thread = client.beta.threads.create()



def main(query):
    # Simulate the message creation process
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query,
    )

    # Simulate the run creation process
    run = client.beta.threads.runs.create(   
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as User"
    )
    
    while True:
        # Wait for 5 seconds
        time.sleep(5)

        # Simulate retrieving the run status
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        # Check if the run is completed
        if run_status.status == "completed":
            # Simulate retrieving messages
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )

            response = {"text": "", "image": None}

            # Process each message
            for message in messages.data:
                if message.role == "assistant":  # Accessing as an attribute
                    for content_block in message.content:
                        if content_block.type == "text":
                            response["text"] += content_block.text.value + "\n\n"
                        elif content_block.type == "image_file":
                            # Assuming you have a way to access the image via its file ID
                            image_url = client.beta.files.get_url(content_block.image_file.file_id)
                            response["image"] = image_url
            
            # Return a tuple of text and image
            return response["text"], response["image"]

        else:
            continue

# Gradio Interface to handle both text and image
iface = gr.Interface(
    fn=main, 
    inputs="text", 
    outputs=["text", "image"], 
    title="Math Tutor",
)
iface.launch()
                    