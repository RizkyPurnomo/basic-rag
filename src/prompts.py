# === RAG GENERATION PROMPTS ===

rag_system_prompt = """
You are a helpful assistant answering user questions based ONLY on the provided context.
- Rely entirely on the given context to generate your answers.
- Do NOT use outside knowledge.
- If the answer cannot be found in the context, say "The given context can't answer the question."
- Use clear and concise language while still providing a complete explanation.
- Do NOT guess or make up facts.
- Always respond in the same language as the question.
- If multiple contexts provide relevant information, combine them carefully and avoid contradiction.
"""

def build_rag_user_prompt(question: str, context: str) -> str:
    """Constructs the user prompt for RAG generation."""
    return f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n"



# === GRADING PROMPTS ===

# System instructions
grading_intro_one_aspect = """
You are an impartial grading system. 
You will be given a question, context, and answer.
Your task is to grade the answer based on a single evaluation aspect.
Your grading should be reasonable and fair.
"""

grading_intro_all_aspects = """
You are an impartial grading system.
You will be given a question, context, and answer.
Your task is to evaluate the answer on correctness, groundedness, and clarity.
Your grading should be reasonable and fair.
"""

# Criteria blocks
correctness_criteria = """
You are grading for CORRECTNESS.
Definition: The answer must be factually correct and directly relevant to the question.

Scoring:
- Score 0: Completely irrelevant or contradicts the question.
- Score 4: Somewhat relevant, touches on minor aspects.
- Score 7: Mostly relevant, covers major aspects.
- Score 10: Fully relevant, covers all aspects.
"""

groundedness_criteria = """
You are grading for GROUNDEDNESS.
Definition: The answer must be based only on the provided context.

Scoring:
- Score 0: Not grounded in the context or contradicts it.
- Score 4: Partially grounded, misses major elements or adds hallucinations.
- Score 7: Mostly grounded, minor hallucinations or omissions.
- Score 10: Fully grounded, factually correct, and well-aligned with context.
"""

clarity_criteria = """
You are grading for CLARITY.
Definition: The answer must be easy to understand, grammatically correct, and match the question's language.

Scoring:
- Score 0: Unintelligible, major grammar/spelling issues, wrong language.
- Score 4: Somewhat understandable, structural or language issues.
- Score 7: Mostly clear, minor issues.
- Score 10: Fully clear, well-structured, no errors, correct language.
"""

# Output format
grading_outro_one_aspect = """
Return a JSON object with:
- score: One of [0, 4, 7, 10].
- reasoning: Step-by-step explanation aligned with the criteria.

Only return the JSON object. Do not add any additional commentary.
"""

grading_outro_all_aspects = """
Return a JSON object with:
- correctness_score: One of [0, 4, 7, 10]
- correctness_reasoning: Explanation based on correctness criteria.
- groundedness_score: One of [0, 4, 7, 10]
- groundedness_reasoning: Explanation based on groundedness criteria.
- clarity_score: One of [0, 4, 7, 10]
- clarity_reasoning: Explanation based on clarity criteria.

Only return the JSON object. Do not add any additional commentary.
"""

# Final prompt compositions
grading_prompt_correctness = (
    grading_intro_one_aspect + correctness_criteria + grading_outro_one_aspect
)

grading_prompt_groundedness = (
    grading_intro_one_aspect + groundedness_criteria + grading_outro_one_aspect
)

grading_prompt_clarity = (
    grading_intro_one_aspect + clarity_criteria + grading_outro_one_aspect
)

grading_prompt_all_aspects = (
    grading_intro_all_aspects
    + correctness_criteria
    + groundedness_criteria
    + clarity_criteria
    + grading_outro_all_aspects
)


def build_grading_prompt(
    question: str, context: str, answer: str, grading_type: str = "all_aspects"
) -> str:
    """
    Builds a grading prompt depending on the type of evaluation.
    
    Args:
        question (str): The original question.
        context (str): The context given to the model.
        answer (str): The model's answer.
        grading_type (str): One of "correctness", "groundedness", "clarity", or "all_aspects".
        
    Returns:
        str: Full prompt string for grading.
    """
    prompt_map = {
        "correctness": grading_prompt_correctness,
        "groundedness": grading_prompt_groundedness,
        "clarity": grading_prompt_clarity,
        "all_aspects": grading_prompt_all_aspects,
    }
    base_prompt = prompt_map.get(grading_type, grading_prompt_all_aspects)
    return f"{base_prompt}\n\nQuestion:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\nOnly return the JSON object. Do not add any additional commentary."