import re
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class StopAtTrueOrFalse(StoppingCriteria):
    def __init__(self, stop_phrases, tokenizer, init_len):
        super().__init__()
        self.stop_phrases = stop_phrases
        self.tokenizer = tokenizer
        self.init_len = init_len

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the generated tokens so far
        generated_text = self.tokenizer.decode(input_ids[0][self.init_len:], skip_special_tokens=True)
        # Check if any of the stop phrases are in the generated text
        for phrase in self.stop_phrases:
            if phrase in generated_text:
                return True
        return False

class StopAtFinalAnswer(StoppingCriteria):
    def __init__(self, tokenizer, init_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.init_len = init_len

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the generated text so far
        generated_text = self.tokenizer.decode(input_ids[0][self.init_len:], skip_special_tokens=True)
        
        if (re.search(r"Final Answer:\s*\d+(\.(\d){2})+", generated_text)) or re.search(r"\s*\d+(\.(\d){2})+", generated_text):
            return True
        return False

def prepare_input(prompt, model_name, tokenizer):
    if 'qwen' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        inputs = tokenizer([text], return_tensors="pt").to('cuda')
    else:
        inputs = tokenizer(prompt, return_tensors = 'pt').to('cuda')
    return inputs

def greedy_generation(model, model_name, tokenizer, inputs, prompt, max_tokens=32):
    # if 'qwen' in model_name.lower():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=max_tokens,
    #         return_dict_in_generate=True,
    #         output_hidden_states=True,
    #         )
    # else:
    if 'verb' in prompt:
        stop = 'numeric'
    else:
        stop = 'true_false'
    if stop=='true_false':
        # Define the stopping phrases
        stop_phrases = ["The answer is true.", "The answer is false.", "Therefore, the answer is true.", "Therefore, the answer is false.", "Therefore, the statement is true.", "Therefore, the statement is false.", "is false.", "is true.", "TRUE.", "FALSE.", "TRUE!", "FALSE!"]
        stopping_criteria = StoppingCriteriaList([StopAtTrueOrFalse(stop_phrases, tokenizer, inputs['input_ids'].shape[1])])
    elif stop=='numeric':        
        stopping_criteria = StoppingCriteriaList([StopAtFinalAnswer(tokenizer, inputs['input_ids'].shape[1])])
    
    outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
    
    embeddings = outputs.hidden_states[-1]#[-1].squeeze().cpu()
    return outputs, embeddings

def saplama_hidden_states(model, inputs):
    with torch.no_grad():
        outputs = model(inputs.input_ids, return_dict_in_generate=True, output_hidden_states=True)
    hs_tuples = outputs["hidden_states"]
    return hs_tuples

def decode(model_name, inputs, outputs, tokenizer):
    
    if 'qwen' in model_name:
        generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs.sequences)
            ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        response = tokenizer.decode(outputs.sequences[0][inputs['input_ids'].shape[1]:]).strip()
    return response