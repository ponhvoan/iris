import pickle
import numpy as np
import torch
from datasets import load_dataset

def format_qa(question, choice):
    return f"Q: {question} A: {choice}"

def format_prompt(dataset_name, dataset, prompt_type): 
    all_prompts = []
    all_labels = []

    if dataset_name == 'halueval':
        for i in range(len(dataset)):
            responses= dataset[i]['chatgpt_fact']
            labels = dataset[i]['human_judge']
            
            if len(responses) == len(labels):
                for j in range(len(responses)): 
                    choice = responses[j]
                    label = labels[j]
                    with open(f'prompts/{prompt_type}.txt', "r", encoding="utf-8") as f:
                        prompt = f.read()
                    prompt = prompt.format(query=choice)
                    all_prompts.append(prompt)
                    all_labels.append(label)
        
    elif dataset_name in ['true_false', 'helm']:
        for i in range(len(dataset)):
            label = int(dataset[i]['label'])
            ans = dataset[i]['statement']
            with open(f'prompts/{prompt_type}.txt', "r", encoding="utf-8") as f:
                prompt = f.read()
            prompt = prompt.format(query=ans)
            all_prompts.append(prompt)
            all_labels.append(label)
            
    return all_prompts, all_labels

def prepare_dataset(dataset_name, topic):
    if dataset_name == 'halueval':
        dataset = load_dataset('json', data_files=f'data/halueval_data/{topic}.json')['train']
        dataset = dataset.map(lambda x: {"human_judge": [1 if val.lower() == "true" else 0 for val in x["human_judge"]]})
    elif dataset_name in ['true_false', 'helm']:
        dataset = load_dataset('csv', data_files=f'data/{dataset_name}_data/{topic}.csv')['train']
        
    formatter = format_prompt
    return dataset, formatter 

def load_generations(args):
    '''
    layer: embeddings from which layer to use, indexed from the last; if 'layer'=='all', use all layers
    unc_type: which metric of uncertainty to use as soft labels
    '''
    dataset_name = args.dataset_name
    topic = args.topic
    model_name = args.model
    prompt = args.prompt
    layer = args.layer
    pseudo_label= args.pseudo_label
    gen_path = f'features/{dataset_name}/{prompt}/{topic}/{model_name.split('/')[-1]}_generations.pkl'
        
    # Load generations
    with open(gen_path, 'rb') as f:
        generations = pickle.load(f)

    emb = []
    for i in range(len(generations)):
        if layer == 'all':
            em = torch.stack(generations[i]['hidden_states'][1:]).squeeze().cpu()
            # em = generations[i]['hidden_states'][1:]
            # em = torch.stack([e.squeeze() for e in em]).cpu()
        else:
            em = generations[i]['hidden_states'][-int(layer)].squeeze().cpu()
        label = generations[i]['label']
        emb.append(em)
    emb = np.array(emb)
    
    # Load labels and soft pseudolabels
    label_path = gen_path
    with open(label_path, 'rb') as f:
        all_labels = pickle.load(f)
        
    soft_labels, labels = [], []
    for i in range(len(all_labels)):
        try:
            all_labels[i]['soft_label'][pseudo_label]
            # all_labels[i]['label_llm']
        except KeyError:
            continue
        label = all_labels[i]['label']
        if pseudo_label in ['entropy', 'perp']:
            soft_label = all_labels[i]['soft_label'][pseudo_label].item()
        else:
            soft_label = all_labels[i]['soft_label'][pseudo_label]
        soft_labels.append(soft_label)
        labels.append(label)
        
    soft_labels = np.array(soft_labels)
    labels = np.array(labels)
    # soft_labels = 1 - (soft_labels/soft_labels.max())
    # z = (soft_labels - np.mean(soft_labels))/np.std(soft_labels)
    # confidence = 1 / (1 + np.exp(z))
    
    return emb, soft_labels, labels