from zhipuai import ZhipuAI
from openai import OpenAI
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
from typing import List, Tuple
import numpy as np
from .text_processor import split_into_sentences


class ZhipuAIClient:
    """
    A class to handle ZhipuAI API interactions
    """
    
    def __init__(self, api_key: str):
        self.client = ZhipuAI(api_key=api_key)
        self.cot_prompt = """
Please describe the entire process step by step. Do not use any lists or markdown formatting. Provide a concise narrative only.
"""

    def generate_text(self, question: str) -> str:
        """
        Generate text using ZhipuAI
        """
        response = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "user", "content": self.cot_prompt},  # First provide Chain of Thought (CoT) prompt
                {"role": "user", "content": question}    # Then provide specific question
            ]
        )
        return response.choices[0].message.content

    def choose_sentences_for_modification_by_labels(self, sentences: List[str], labels: List[int]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Choose sentences for modification based on professionalism
        """
        print("\nAll sentences and their label information:")
        # Output each sentence's number, content, and label information
        for idx, (sentence, label) in enumerate(zip(sentences, labels)):
            print(f"Sentence {idx + 1} (Label: {label}): {sentence}")
        
        # Generate a prompt asking the model to judge the professionalism of each sentence
        prompt = "Please evaluate the following sentences and identify the sentence that deviates the most from professional standards. Output the content of that sentence and its label.\n" + "\n".join([f"{i + 1}. {sentence} (Label: {label})" for i, (sentence, label) in enumerate(zip(sentences, labels))])

        # Send request to ZhipuAI, explicitly calling the large language model
        response = self.client.chat.completions.create(
            model="glm-4-flash",  # Use appropriate model, large language model (please choose model version as needed)
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the least professional sentence content from response
        response_text = response.choices[0].message.content.strip()
        least_professional_sentence, least_professional_label = None, None

        # Extract sentence and label
        for line in response_text.split("\n"):
            line = line.strip()
            for idx in range(len(sentences)):
                if line.startswith(f"{idx + 1}."):
                    least_professional_sentence = sentences[idx]
                    least_professional_label = labels[idx]
                    break
            if least_professional_sentence:
                break

        # Fallback if least professional sentence not found
        if least_professional_sentence is None:
            return [], []

        # Add least professional sentence to sentences needing modification, classify remaining sentences as not needing modification
        sentences_to_modify = [(least_professional_sentence, least_professional_label)]  # Include label
        sentences_not_to_modify = [(sentence, label) for sentence, label in zip(sentences, labels) if sentence != least_professional_sentence]

        return sentences_to_modify, sentences_not_to_modify


class DeepSeekClient:
    """
    A class to handle DeepSeek API interactions
    """
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def generate_expansion(self, sentences_to_modify: List[Tuple[str, int]], cot_prompt: str, previous_generations: List[List[str]] = None) -> str:
        """
        Generate text expansion using DeepSeek
        """
        additional_instruction = "Please ensure that the generated expanded sentences maintain as high similarity as possible to the original sentence and ensure semantic consistency."
        detailed_prompt = cot_prompt + "\n" + "Please provide a detailed explanation and expansion of the following sentences. Use words from the sentence to ensure that the sentence's meaning remains unchanged and ensure high similarity to the original sentence. The output should contain at least two sentences but no more than three sentences:" + "\n" + "\n".join([sentence for sentence, _ in sentences_to_modify]) + "\n" + additional_instruction
        
        # Add historical prompts when regeneration is required
        if previous_generations:
            avoid_prompt = "\nPlease avoid repeating the following expressions:\n" + "\n".join([f"- {sentence}" for gen in previous_generations for sentence in gen]) + "\nPlease generate new content that is more semantically similar to the original sentence."
            detailed_prompt += avoid_prompt

        # Send request to DeepSeek
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": detailed_prompt}
            ]
        )
        
        return response.choices[0].message.content


class GPT2PerplexityCalculator:
    """
    A class to calculate perplexity using GPT-2 model
    """
    
    def __init__(self, model_name: str = 'gpt2'):
        self.model_name = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.eval()

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity for a given text
        """
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            log_likelihood = outputs.loss.item()
        perplexity = math.exp(log_likelihood)
        return perplexity

    def check_sentence_quality(self, sentences: List[str], max_perplexity: float = 500) -> bool:
        """
        Check if all sentences meet the perplexity requirement
        """
        for sentence in sentences:
            perplexity = self.calculate_perplexity(sentence)
            if perplexity > max_perplexity:
                return False
        return True