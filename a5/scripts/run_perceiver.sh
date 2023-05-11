#! /bin/bash

# Pretrain the model
python src/run.py pretrain perceiver wiki.txt --bottleneck_dim 64 \
        --pretrain_lr 6e-3 --writing_params_path perceiver.pretrain.params
        
# Finetune the model
python src/run.py finetune perceiver wiki.txt --bottleneck_dim 64 \
        --reading_params_path perceiver.pretrain.params \
        --writing_params_path perceiver.finetune.params \
        --finetune_corpus_path birth_places_train.tsv
        
# Evaluate on the dev set; write to disk
python src/run.py evaluate perceiver wiki.txt --bottleneck_dim 64 \
        --reading_params_path perceiver.finetune.params \
        --eval_corpus_path birth_dev.tsv \
        --outputs_path perceiver.pretrain.dev.predictions
        
# Evaluate on the test set; write to disk
python src/run.py evaluate perceiver wiki.txt --bottleneck_dim 64 \
        --reading_params_path perceiver.finetune.params \
        --eval_corpus_path birth_test_inputs.tsv \
        --outputs_path perceiver.pretrain.test.predictions
