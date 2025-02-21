## Unsupervised Hallucincation Detection by Inspecting Reasoning Processes

We are interested in unsupervised hallucination detection using only a proxy LLM internal states. 
The detection pipeline works in two stages:
1. For each statement in the dataset, we prompt the proxy model to carefully reason whether it is true or false. We obtain its internal activations and uncertainty in its response as pseudolabel.
2. We train a binary classifier to determine whether a statement is hallucinated using the model activations as features and the pseudolabels.

### Get Started

Create a conda environment and install requirements.

```
conda create -n iris python=3.12
conda activate iris
pip install -r requirements.txt
```

Generate embeddings.
```
python generate_embeddings.py
```

Then, train classifier. Hyperparameters may need finetuning.
```
python train_classifier.py
```

