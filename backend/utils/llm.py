from google import genai
import logging
from dotenv import load_dotenv
import os
import re
from pydantic import BaseModel, field_validator, ValidationError
from typing import Union, Literal
from models import AnswerFormat

load_dotenv()

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini client and model constants
MODEL_NAME = "gemini-2.0-flash"
TOKEN_SAFETY_BUFFER = 200 # Buffer for variations and safety
BASE_TOKEN_BUDGET_PER_DOC = 5000

# Pydantic Models for Validation
class DateResponse(BaseModel):
    answer: str

    @field_validator('answer')
    @classmethod
    def check_date_format(cls, v: str) -> str:
        # ISO-8601 YYYY-MM-DD
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

class BooleanResponse(BaseModel):
    answer: Literal['Yes', 'No']

class CurrencyResponse(BaseModel):
    answer: str

    @field_validator('answer')
    @classmethod
    def check_currency_format(cls, v: str) -> str:
        # Matches patterns like "500 SEK", "30 USD", "123.45 EUR"
        if not re.match(r"^\d+(\.\d{1,2})?\s[A-Z]{3}$", v):
            raise ValueError("Currency must be in format 'AMOUNT CURRENCY_CODE' (e.g., '500 SEK', '30.50 USD')")
        return v

# Mapping formats to Pydantic models
PYDANTIC_MODELS = {
    "date": DateResponse,
    "boolean": BooleanResponse,
    "currency": CurrencyResponse,
    "text": None  # No specific Pydantic model for free-form text
}

PROMPT = """
You are a helpful assistant that can answer questions about the context provided.

{context}

Question: {question}

The answer should be:
{format_instruction}
"""

FORMAT_INSTRUCTIONS = {
    "text": "Free-form text.",
    "date": "A date in ISO-8601 format (e.g., YYYY-MM-DD). For example: 2023-10-26",
    "boolean": "A definite 'Yes' or 'No', and nothing else. For example: Yes",
    "currency": "A currency value including the amount and currency code (e.g., 500 SEK, 30 USD). For example: 125.99 USD"
}

def _call_llm(prompt_formatted: str) -> str:
    logger.debug(f"Sending prompt to LLM (first 200 chars): {prompt_formatted[:200]}...")
    response = client.models.generate_content(
        model=MODEL_NAME, contents=prompt_formatted
    )
    return response.text.strip() if response.text else ""

def get_answer(prompt: str, context: str, format: Union[str, AnswerFormat], n_documents: int = 1, retries: int = 1) -> str:
    logger.info(f"Getting answer for prompt: '{prompt[:50]}...' with format: {format}, n_docs: {n_documents}")

    format_key = format.value if isinstance(format, AnswerFormat) else format

    base_format_instruction = FORMAT_INSTRUCTIONS.get(format_key)
    if not base_format_instruction:
        logger.error(f"Invalid format specified: {format_key}. Defaulting to text.")
        base_format_instruction = FORMAT_INSTRUCTIONS["text"]
        format_key = "text"
    
    llm_response_text = ""

    current_input_token_budget = BASE_TOKEN_BUDGET_PER_DOC * n_documents
    logger.info(f"Current total input token budget based on n_documents ({n_documents}): {current_input_token_budget}")

    for attempt in range(retries + 1):
        current_format_instruction = base_format_instruction
        if attempt > 0:
            logger.warning(f"Retry {attempt}/{retries} for prompt: '{prompt[:50]}...' due to validation failure.")
            current_format_instruction = f"Ensure the answer STRICTLY follows this format: {base_format_instruction}. Previous attempt failed validation."

        # get tokens for prompt structure (without actual context)
        prompt_structure_template = PROMPT.format(context="{CONTEXT_PLACEHOLDER}", question=prompt, format_instruction=current_format_instruction)
        tokens_for_prompt_structure = client.models.count_tokens(
            model=MODEL_NAME, 
            contents=prompt_structure_template.replace("{CONTEXT_PLACEHOLDER}", "")
        ).total_tokens

        max_tokens_for_context = current_input_token_budget - tokens_for_prompt_structure - TOKEN_SAFETY_BUFFER
        logger.info(f"Tokens for prompt structure: {tokens_for_prompt_structure}. Max tokens available for context: {max_tokens_for_context}")

        context_formatted_for_llm = context # Start with the full original context for this attempt

        if max_tokens_for_context <= 0:
            logger.error("Prompt structure and safety buffer exceed total token budget, even with no context. Using empty context.")
            context_formatted_for_llm = ""
        else:
            current_context_tokens = client.models.count_tokens(model=MODEL_NAME, contents=context_formatted_for_llm).total_tokens
            
            if current_context_tokens > max_tokens_for_context:
                logger.warning(f"Context ({current_context_tokens} tokens) exceeds max allowed for context ({max_tokens_for_context} tokens). Performing token-based truncation.")
                # truncate iteratively until token count is met.
                while current_context_tokens > max_tokens_for_context and len(context_formatted_for_llm) > 0:
                    chars_to_cut = max(1, int(len(context_formatted_for_llm) * 0.1))
                    context_formatted_for_llm = context_formatted_for_llm[:-chars_to_cut]
                    if not context_formatted_for_llm: 
                        break
                    current_context_tokens = client.models.count_tokens(model=MODEL_NAME, contents=context_formatted_for_llm).total_tokens
                logger.info(f"Context truncated to {current_context_tokens} tokens and {len(context_formatted_for_llm)} chars.")
            else:
                logger.info(f"Context ({current_context_tokens} tokens) fits within max allowed for context ({max_tokens_for_context} tokens).")

        current_prompt_to_llm = PROMPT.format(context=context_formatted_for_llm, question=prompt, format_instruction=current_format_instruction)
        
        llm_response_text = _call_llm(current_prompt_to_llm)
        logger.info(f"LLM Response (attempt {attempt + 1}): {llm_response_text}")

        validator_model = PYDANTIC_MODELS.get(format_key)
        if validator_model:
            try:
                validated_data = validator_model(answer=llm_response_text)
                logger.info(f"Validation successful for format '{format_key}' with response: {validated_data.answer}")
                return validated_data.answer
            except ValidationError as ve:
                logger.warning(f"Validation failed for format '{format_key}' on attempt {attempt + 1}: {ve}. Response: '{llm_response_text}'")
                if attempt >= retries:
                    logger.error(f"Final validation failed after {retries} retries for format '{format_key}'. Returning raw response.")
                    return llm_response_text
        else: # For "text" format or if no validator is defined
            logger.info(f"No Pydantic validator for format '{format_key}'. Returning raw response.")
            return llm_response_text

    return llm_response_text # Fallback, should ideally be covered by logic above