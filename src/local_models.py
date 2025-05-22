import requests
import json
import re 

from prompts import rag_system_prompt, build_rag_user_prompt, build_grading_prompt

OLLAMA_URL = "http://localhost:11434/api/generate"

def call_ollama(system_prompt: str = "You are a helpful assistant", user_prompt: str = "", model: str = 'llama3.1:8b') -> str:
    """
    Sends a chat-style prompt to a locally running Ollama model.

    :param model: The model name (e.g., 'llama3.1:8b' or 'deepseek-r1:1.5b').
    :param system_prompt: The system instruction for the assistant.
    :param user_prompt: The actual user input (can include context, question, answer, etc.).
    :return: The model's generated text.
    """
    payload = {
        "model": model,
        "system": system_prompt.strip(),
        "prompt": user_prompt.strip(),
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"].strip()


def extract_response_from_deepseekr1(response_str):
    return re.sub(r"<think>.*?</think>", "", response_str, flags=re.DOTALL).strip()

def extract_json_grade_from_deepseekr1_response(response_str):
    match = re.search(r'\{[\s\S]*\}', response_str)
    if match:
        json_str = match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("❌ Failed to parse extracted JSON:\n", json_str)


def rag_generate_ollama(question: str, context: str, model: str = "llama3.1:8b") -> str:
    """
    Generate an answer to the question using the provided context via Ollama model.

    :param question: The user question.
    :param context: The retrieved documents or context.
    :param system_prompt: The system instruction to guide the model.
    :param model: The Ollama model to use for generation.
    :return: The generated answer.
    """
    system_prompt = rag_system_prompt
    user_prompt = build_rag_user_prompt(question, context)
    return call_ollama(system_prompt=system_prompt, user_prompt=user_prompt, model=model)

def grade_answer_ollama(question: str, context: str, answer: str, model: str = "deepseek-r1:1.5b") -> dict:
    """
    Grade the generated answer using a separate grading Ollama model.

    :param question: The original user question.
    :param context: The context that was used for generation.
    :param answer: The generated answer.
    :param grading_prompt: The full grading prompt (system + criteria).
    :param model: The Ollama model to use for grading.
    :return: A parsed JSON dict with grading scores and reasoning.
    """
    user_prompt = build_grading_prompt(question, context, answer)
    response = call_ollama(user_prompt=user_prompt, model=model)
    if model == 'deepseek-r1:1.5b':
        response = extract_json_grade_from_deepseekr1_response(response)
    return response
