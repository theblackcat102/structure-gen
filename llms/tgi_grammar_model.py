import os
import json
import logging
import torch
import pydantic_core
from huggingface_hub import InferenceClient

class TGI():

    def __init__(self, model_name) -> None:
        # bad docs
        self.client = InferenceClient(os.environ["TGI_ENDPOINT"])
        self.model_name = model_name

    def __call__(self, prompt, schemas, max_tokens=512, temperature=0.0, **kwargs) -> str:
        # https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/using_guidance


        # weird behavior
        # f"{prompt} please use the following schema: {schemas.schema()}"
        # prompt
        # > { "answer": 0, "reasoning": "" }
        for _ in range(5):
            try:
                event = self.client.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.001), # must be strictly positive
                    grammar={"type": "json", "value": schemas.schema()},
                )
                event = json.loads(json.dumps(event)) # make sure its really JSON
                success = True
                break
            except (json.decoder.JSONDecodeError, AssertionError) as e:
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
    llm = TGI('meta-llama/Meta-Llama-3-8B-Instruct')
    res, res_info = llm(prompt="Follow the instruction to complete the task:\nMathematical problem-solving task:\n• Given: A mathematical question or problem\n• Required: A numerical answer only\n• Role: You are a math tutor assisting students of all levels\n• Process: Think step by step to solve the problem\nNote: Read the question carefully before beginning your analysis.\n\n\nInstruct : Provide your output in the following valid JSON format:\n```json\n{\n    \"reason\": ...,\n    \"answer\": ...\n}\n```\n\n\n\nQuestion: Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
    schemas=Response)
    print(res)
