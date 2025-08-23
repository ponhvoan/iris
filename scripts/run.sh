MODELS='meta-llama/Llama-3.2-3B-Instruct'
DATASETS='true_false halueval'
PROMPT='cot'
PSEUDO_LABEL='verb'

for MODEL in $MODELS
do
    for DATA in $DATASETS
    do
        if [ $DATA = 'true_false' ]; then
            TOPICS='animals cities companies elements facts generated inventions'
        elif [ $DATA = 'halueval' ]; then
            TOPICS='Bio-Medical Education Finance Open-Domain Science'
        elif [ $DATA = 'helm' ]; then
            TOPICS='falcon40b gptj7b llamabase7b llamachat7b llamachat13b opt7b'
        fi

        for TOPIC in $TOPICS
            do
                python generate_embeddings.py --model $MODEL --dataset_name $DATA --topic $TOPIC --prompt $PROMPT --save_generation --verb
                python train_classifier.py --model $MODEL --dataset_name $DATA --topic $TOPIC --classifier mlp --num_epochs 10 --prompt $PROMPT --layer 1 --pseudo_label $PSEUDO_LABEL --beta 0.8 --phi 1 --normalize
            done
    done
done
