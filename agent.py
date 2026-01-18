from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory

# Load Hugging Face model (FREE)
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)
memory = ConversationBufferMemory()

print("🎓 AI Exam Preparation System (Hugging Face – Free)")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    memory.save_context({"input": user_input}, {"output": ""})

    if "mcq" in user_input.lower():
        prompt = f"Generate 5 MCQs with answers from the topic: {user_input}"
    elif "important" in user_input.lower():
        prompt = f"Generate 5 important exam questions from: {user_input}"
    elif "explain" in user_input.lower():
        prompt = f"Explain the following concept in simple words: {user_input}"
    elif "revision" in user_input.lower():
        prompt = f"Give last day exam revision tips for: {user_input}"
    else:
        prompt = user_input

    response = llm(prompt)
    print("Agent:", response)
