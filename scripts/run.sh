MODELS='meta-llama/Llama-3.2-1B-Instruct'
DATASETS='helm'
PROMPT='cot'

for MODEL in $MODELS
do
    for DATA in $DATASETS
    do
        if [ $DATA = 'true_false' ]; then
            TOPICS='animals cities companies elements facts generated inventions'
            PSEUDO_LABEL='verb'
        elif [ $DATA = 'halueval' ]; then
            TOPICS='Bio-Medical Education Finance Open-Domain Science'
            PSEUDO_LABEL='verb'
        elif [ $DATA = 'helm' ]; then
            TOPICS='falcon40b gptj7b llamabase7b llamachat7b llamachat13b opt7b'
            PSEUDO_LABEL='entropy'
        fi

        for TOPIC in $TOPICS
            do
                python generate_embeddings.py --model $MODEL --dataset_name $DATA --topic $TOPIC --prompt $PROMPT --save_generation --verb
                python train_classifier.py --model $MODEL --dataset_name $DATA --topic $TOPIC --classifier mlp --num_epochs 10 --prompt $PROMPT --layer 1 --pseudo_label $PSEUDO_LABEL --beta 0.8 --beta2 1 --normalize
            done
    done
done
