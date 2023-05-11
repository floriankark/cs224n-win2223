##! /bin/bash

# Note: Don't forget to edit the hyper-parameters for part d.

# Train on the names dataset
python src/run.py finetune vanilla wiki.txt \
--writing_params_path vanilla.model.params \
--finetune_corpus_path birth_places_train.tsv
# Evaluate on the dev set, writing out predictions
python src/run.py evaluate vanilla wiki.txt \
--reading_params_path vanilla.model.params \
--eval_corpus_path birth_dev.tsv \
--outputs_path vanilla.nopretrain.dev.predictions
# Evaluate on the test set, writing out predictions
python src/run.py evaluate vanilla wiki.txt \
--reading_params_path vanilla.model.params \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path vanilla.nopretrain.test.predictions
