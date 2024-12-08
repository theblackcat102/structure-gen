"""
docker run --gpus 1 \
        --shm-size 32g \
        -p 30000:30000 \
         -v your_cache_path/hf_cache:/root/.cache/huggingface \
         --env "HF_TOKEN=XXXX" \
        --ipc=host \
        lmsysorg/sglang:latest \
        python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --host 0.0.0.0 --port 30000
>> pip install outlines==0.0.46
"""
import os
import json
from enum import Enum
import sglang as sgl
from sglang.srt.constrained import build_regex_from_object
from transformers import AutoTokenizer
sgl.set_default_backend(sgl.RuntimeEndpoint(os.environ["SGLANG_ENDPOINT"]))

@sgl.function
def pydantic_wizard_gen(s, prompt, schema, max_tokens):
    s += prompt
    s += sgl.gen(
        "json_output",
        max_tokens=max_tokens,
        temperature=0, # since we are using temp=0 all the time, skip
        regex=build_regex_from_object(schema),  # Requires pydantic >= 2.0
    )

class SGLang():

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def __call__(self, prompt, schemas, max_tokens=512, temperature=0.0, **kwargs) -> str:
        messages = [
            {"role": "user", "content": prompt},
        ]
        reformulated_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        success = False
        assistant_start = reformulated_prompt.strip()[-20:]
        # in theory this is deterministic which means rerun doesn't change the result
        for _ in range(5):
            try:
                state = pydantic_wizard_gen.run(prompt=reformulated_prompt, schema=schemas, max_tokens=max_tokens)
                result = state.text()
                response = result.split(assistant_start, maxsplit=1)[-1].strip()
                event = json.loads(response)
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
            "output": response,
        }
        return event, res_info

if __name__ == "__main__":
    sgl.set_default_backend(sgl.RuntimeEndpoint(os.environ["SGLANG_ENDPOINT"]))
    from pydantic import BaseModel
    class Response(BaseModel):
        reasoning: str
        answer: int
    llm = SGLang('google/gemma-2-9b-it')
    res, res_info = llm(prompt="Follow the instruction to complete the task:\nMathematical problem-solving task:\n• Given: A mathematical question or problem\n• Required: A numerical answer only\n• Role: You are a math tutor assisting students of all levels\n• Process: Think step by step to solve the problem\nNote: Read the question carefully before beginning your analysis.\n\n\nInstruct : Provide your output in the following valid JSON format:\n```json\n{\n    \"reason\": ...,\n    \"answer\": ...\n}\n```\n\n\n\nQuestion: Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
        schemas=Response)
    print(res)