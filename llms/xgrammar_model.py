import os
import json
import logging
import torch
import pydantic_core
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import xgrammar as xgr

class XGrammar():

    def __init__(self, model_name) -> None:
        # bad docs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.compiled_grammar = None
        full_vocab_size = self.config.vocab_size
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=full_vocab_size)

        self.compiler = xgr.GrammarCompiler(self.tokenizer_info, max_threads=8)
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                            device_map='cuda' if torch.cuda.is_available() else 'cpu',
                                                            torch_dtype=torch.bfloat16)
        self.model.eval()

    def __call__(self, prompt, schemas, max_tokens=512, temperature=0.0, **kwargs) -> str:
        if self.compiled_grammar is None:
            self.compiled_grammar = self.compiler.compile_json_schema(schemas)
        
        messages = [
            {"role": "user", "content": prompt},
        ]
        texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(texts, return_tensors="pt").to(self.model.device)
        xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)
        success = False
        # in theory this is deterministic which means rerun doesn't change the result
        for _ in range(5):
            try:
                generated_ids = self.model.generate(
                    **model_inputs, max_new_tokens=max_tokens, logits_processor=[xgr_logits_processor],
                    temperature=temperature,
                    do_sample=False
                )
                generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
                outputs = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                event = json.loads(outputs)
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
    llm = XGrammar('meta-llama/Meta-Llama-3-8B-Instruct')
    res, res_info = llm(prompt="Follow the instruction to complete the task:\nMathematical problem-solving task:\n• Given: A mathematical question or problem\n• Required: A numerical answer only\n• Role: You are a math tutor assisting students of all levels\n• Process: Think step by step to solve the problem\nNote: Read the question carefully before beginning your analysis.\n\n\nInstruct : Provide your output in the following valid JSON format:\n```json\n{\n    \"reason\": ...,\n    \"answer\": ...\n}\n```\n\n\n\nQuestion: Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?",
    schemas=Response)
    print(res)
