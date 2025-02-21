import os
import re
import argparse
import pickle

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from utils.dataset import  prepare_dataset
from utils.generate import prepare_input, greedy_generation, decode
from utils.losses import compute_soft_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, default='helm', choices=['halueval', 'true_false', 'helm'])
    parser.add_argument('--topic', type=str, default='falcon40b')
    parser.add_argument('--prompt', type=str, default='cot')
    parser.add_argument('--verb', action='store_true', default=True)
    parser.add_argument('--save_generations', action='store_true')
    args = parser.parse_args()

    if args.dataset_name == 'halueval' and args.topic not in ['Bio-Medical', 'Education', 'Finance', 'Open-Domain', 'Science']:
        parser.error('For halueval dataset, topic must be one of Bio-Medical, Education, Finance, Open-Domain, or Science')
    elif args.dataset_name == 'true_false' and args.topic not in ['animals', 'cities', 'companies', 'elements', 'facts', 'generated', 'inventions']:
        parser.error('For true-false dataset, topic must be one of animals, cities, companies, elements, facts, generated, or inventions')
    elif args.dataset_name == 'helm' and args.topic not in ['falcon40b', 'gptj7b', 'llamabase7b', 'llamachat7b', 'llamachat13b', 'opt7b', 'llama3.2-1b']:
        parser.error('For helm dataset, topic must be one of falcon40b, gptj7b, llamabase7b, llamachat7b, llamachat13b, opt7b')
        
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataset, formatter = prepare_dataset(args.dataset_name, args.topic)
    prompts, labels = formatter(args.dataset_name, dataset, args.prompt)
    if args.verb:    
        prompts_verb, _ = formatter(args.dataset_name, dataset, 'verb')
    labels_llm = []
    excluded = []
    generations = []
    max_tokens = 128 if 'cot' in args.prompt else 32
    
    
    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts), desc='Generating responses'):
        
        inputs = prepare_input(prompt, args.model, tokenizer)
        outputs, embeddings = greedy_generation(model, args.model, tokenizer, inputs, args.prompt, max_tokens)
        response = decode(args.model, inputs, outputs, tokenizer)
        
        if idx==0:
            print(prompt + response)
        
        
        if any(condition in response for condition in ["the answer is true", "it is true", "is true.", "TRUE.", 'TRUE!']):
            label_llm = 1
        else:
            label_llm = 0
        
        # Get pseudo-labels: entropy, perplexity and verbalised confidence
        soft_label_dict = compute_soft_labels(model, outputs, inputs)
        if args.verb:
            inputs_verb = prepare_input(prompts_verb[idx], args.model, tokenizer)
            outputs_verb, _ = greedy_generation(model, args.model, tokenizer, inputs_verb, 'verb', 32)
            verb = decode(args.model, inputs_verb, outputs_verb, tokenizer)
            verb = re.findall(r"\d+\.+\d*", verb)
            if len(verb)>0:
                verb = float(verb[-1])
                soft_label_dict['verb'] = verb
            else:
                excluded.append(idx)
                
        labels_llm.append(label_llm)
        generation_dict = {
            'prompt': prompt,
            'response': response,
            'hidden_states': embeddings,
            'soft_label': soft_label_dict,
            'label_llm': label_llm,
            'label': labels[idx]
        }
        
        generations.append(generation_dict)
    
print(f"Agreement: {sum([labels_llm[i] == labels[i] for i in range(len(labels_llm))])/len(labels_llm)*100:.3f}")
print(f"No response total: {len(excluded)}")

if args.save_generations:
    save_path = f'features/{args.dataset_name}/{args.prompt}/{args.topic}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f'{save_path}/{args.model.split('/')[-1]}_generations.pkl', 'wb') as f:
        pickle.dump(generations, f)
    print("Generations saved")