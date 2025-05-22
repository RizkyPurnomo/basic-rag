import os
import re
import json
from dotenv import load_dotenv
from groq import Groq

from prompts import rag_system_prompt, build_rag_user_prompt, build_grading_prompt

# === Load Environment Variables ===
load_dotenv(dotenv_path="../config.env")

# === Initialize Groq Client ===
client = Groq(api_key=os.getenv("GROQ_API", ""))

# === Model Configuration ===
MODEL_DETAILS = {
    'llama4-scout': {
        'model_name': 'meta-llama/llama-4-scout-17b-16e-instruct',
        'context_length': 128_000,
    },
    'llama4-maverick': {
        'model_name': 'meta-llama/llama-4-maverick-17b-128e-instruct',
        'context_length': 128_000,
    },
    'deepseek-r1': {
        'model_name': 'deepseek-r1-distill-llama-70b',
        'context_length': 128_000,
    },
    'qwen-qwq': {
        'model_name': 'qwen-qwq-32b',
        'context_length': 128_000,
    },
    'mistral-saba': {
        'model_name': 'mistral-saba-24b',
        'context_length': 32_000,
    },
    'llama3.3-70b': {
        'model_name': 'llama-3.3-70b-versatile',
        'context_length': 128_000,
    },
    'llama3.1-8b': {
        'model_name': 'llama-3.1-8b-instant',
        'context_length': 128_000,
    },
}


# === Completion ===
def completion(model_name: str, system_prompt: str = "You are a helpful assistant", user_prompt: str = "") -> str:
    """
    Generate a completion response using the specified model.

    Args:
        model_name (str): Full model name from Groq.
        system_prompt (str): Instruction for the assistant.
        user_prompt (str): User input or constructed RAG prompt.

    Returns:
        str: The assistant's response content.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return response.choices[0].message.content


# === Post-processing (DeepSeek-specific) ===
def extract_response_from_deepseekr1(response_str: str) -> str:
    """
    Remove <think> tags and return cleaned response from DeepSeek-R1.

    Args:
        response_str (str): Raw model output.

    Returns:
        str: Cleaned response.
    """
    return re.sub(r"<think>.*?</think>", "", response_str, flags=re.DOTALL).strip()


def extract_json_grade_from_deepseekr1_response(response_str: str) -> dict:
    """
    Extract and parse JSON content from DeepSeek-R1 grading response.

    Args:
        response_str (str): Raw model output.

    Returns:
        dict: Parsed JSON grade object.
    """
    cleaned = extract_response_from_deepseekr1(response_str)
    cleaned = cleaned.strip('` \n')
    cleaned = re.sub(r'^json\n?', '', cleaned)
    return json.loads(cleaned)


# === RAG Response Generator ===
def rag_generate(question: str, context: str, model: str = 'llama4-scout') -> str:
    """
    Generate a RAG response using the specified model.

    Args:
        question (str): User's question.
        context (str): Retrieved context.
        model (str): Short model key from MODEL_DETAILS.

    Returns:
        str: Generated answer.
    """
    model_name = MODEL_DETAILS[model]['model_name']
    user_prompt = build_rag_user_prompt(question, context)

    response = completion(
        model_name=model_name,
        system_prompt=rag_system_prompt,
        user_prompt=user_prompt
    )

    if model == 'deepseek-r1':
        response = extract_response_from_deepseekr1(response)

    return response


# === Grading Response Generator ===
def grading(question: str, context: str, answer: str, model: str = 'deepseek-r1') -> dict:
    """
    Use a model to grade an answer based on a question and context.

    Args:
        question (str): Original question.
        context (str): Relevant context.
        answer (str): Model-generated answer.
        model (str): Short model key from MODEL_DETAILS.

    Returns:
        dict: Grading result, parsed as JSON if DeepSeek-R1.
    """
    model_name = MODEL_DETAILS[model]['model_name']
    user_prompt = build_grading_prompt(question, context, answer)

    response = completion(model_name=model_name, user_prompt=user_prompt)

    if model == 'deepseek-r1':
        return extract_json_grade_from_deepseekr1_response(response)

    return response


# === Optional Debug/Test Run ===
if __name__ == "__main__":
    question = "What is Lora and what are its advantages?"
    context = "Fine-tuning enormous language models is prohibitively expensive in terms of the hardware required and the storage/switching cost for hosting independent instances for different tasks. We propose LoRA, an efficient adaptation strategy that neither introduces inference latency nor reduces input sequence length while retaining high model quality. Importantly, it allows for quick task-switching when deployed as a service by sharing the vast majority of the model parameters. While we focused on Transformer language models, the proposed principles are generally applicable to any neural networks with dense layers."
    response = rag_generate(question, context)
    print("Question:", question)
    print("Context:", context)
    print("RAG Response:", response)

    grade = grading(question, context, response)
    print("Grading Result:", grade)
