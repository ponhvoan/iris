## Unsupervised Hallucincation Detection by Inspecting Reasoning Processes

This repository contains the implementation of Inernal Reasoning for Inference of Statement veracity (IRIS), an unsupervised hallucination detection method that inspects the uncertainty of the reasoning process.
 
The detection pipeline works in two stages:
1. For each statement in the dataset, we prompt the proxy model to carefully reason whether it is true or false. We obtain its the contextualized embeddings, and uncertainty in its response as a pseudolabel.
2. We train a binary classifier to determine whether a statement is hallucinated using the model activations as features, along with the uncertainty pseudolabels.

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

The script `scripts/run.sh` can be used to reproduce experiments in the paper.
