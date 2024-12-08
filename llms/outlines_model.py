import os
import logging
import torch
import pydantic_core
import outlines
from transformers import AutoTokenizer
from outlines.generate.api import SamplingParameters, GenerationParameters
from outlines.samplers import greedy
class OutlinesStructure():

    def __init__(self, model_name) -> None:
        # bad docs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = outlines.models.transformers(model_name, model_kwargs={
            'device_map': 'cuda',
            'torch_dtype': torch.bfloat16
        })
        self.generator = None

    def __call__(self, prompt, schemas, max_tokens=512, temperature=0.0, **kwargs) -> str:
        if self.generator is None:
            sample_params = greedy() # this is slow when additional parsing criteria was added
            self.generator = outlines.generate.json(self.model, schemas, sampler=sample_params)
        success = False
        # in theory this is deterministic which means rerun doesn't change the result
        for _ in range(5):
            try:
                messages = [
                    {"role": "user", "content": prompt},
                ]
                texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                parsed_result = self.generator(texts, max_tokens=max_tokens)
                event = parsed_result.json()
                success = True
                break
            except pydantic_core._pydantic_core.ValidationError as e:
                print(e)
                continue
        if not success:
            event = {
                'answer': 'failed'
            }
        res_info = {
            "input": prompt,
            "output": event,
        }

        return event, res_info



if __name__ == "__main__":
    from pydantic import BaseModel
    class Response(BaseModel):
        reasoning: str
        answer: int
    llm = OutlinesStructure('meta-llama/Meta-Llama-3-8B-Instruct')
    res, res_info = llm(prompt="Follow the instruction to complete the task:\nMathematical problem-solving task:\n• Given: A mathematical question or problem\n• Required: A numerical answer only\n• Role: You are a math tutor assisting students of all levels\n• Process: Think step by step to solve the problem\nNote: Read the question carefully before beginning your analysis.\n\n\nInstruct : Provide your output in the following valid JSON format:\n```json\n{\n    \"reason\": ...,\n    \"answer\": ...\n}\n```\n\n\n\nQuestion: Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
        schemas=Response)
    print(res)
