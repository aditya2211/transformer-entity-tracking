
# Effective Use of Transformer Networks for Entity Tracking (EMNLP19)

This is a PyTorch implementation of our [EMNLP paper]() on the effectiveness of pre-trained transformer architectures in capturing complex entity interaction in procedural texts. 

## Dependencies 

The code was developed by extending Hugging Face's implementations of [OpenAI's GPT](https://github.com/huggingface/pytorch-openai-transformer-lm) and [BERT](https://github.com/huggingface/transformers).

## Dataset and code
The dataset for two tasks: (i) Recipes, and (ii) ProPara can be found [here](https://drive.google.com/file/d/1Y9DUPSiabnBhSoPLLgmGsVE_Gf4if1az/view) in the appropriate directories.

The codebase consists of two main sub-directories:
### `gpt-entity-tracking`
This consist of the codebase for the main ET-GPT model along with the variants, related experimentation, and gradient analysis for the Recipes and ProPara dataset:
* `train_transformer_recipe_lm.py` is the main training code for the Recipes task and following is the example usage:
```
python3 train_transformer_recipe_lm.py --n_iter_lm 5 --n_iter 20 --n_layer 12 --n_head 12 --n_embd 768 --lmval 2000 --lmtotal 50000
```
* `dataset/` folder consists of the complete train/val/test data for the two tasks.
* `save/` folder consists of the saved model params for the best model which can used to reproduce results.
* `log/` folder consists of the training logs after each iteration.
* `run_transformer_recipe_lm.py` load a saved model to perform inference on the test set.
* `train_transformer_recipes_lm5_12_12_768_50000.npy` consists of the probabilities for the test file in dataset folder `test_recipes_task.json`.
* `ingredient_type_annotations_dev_test.json` is the annotated json file containing ground truth whether the ingredient was in a combined or uncombined state in a recipe in a particular time-step. This was file used for calculating Combined Recall and Uncombined Recall.

### `bert-entity-tracking` 
This consists of codebase for the ET-BERT experiments, primarily focused on the ProPara experiments: 

* `bert_propara_context_ing/` and `bert_propara_ing_context/` folders consists of the reproduced results for ProPara experiments. The code for this would be in `bert_propara.py`. 
* `propara_sent_test_bert_et.tsv` consists of the results on the sentence level task and using this [script](https://github.com/allenai/propara/blob/master/propara/evaluation/evalQA.py)
* `propara_sent_val_bert_et.tsv` consists of the results on validation set of sentence level task.
* `para_id.val.txt` and `gold_labels_valid.tsv` are the helper files for val set of ProPara's sentence level task.

## Citation
```
 @inproceedings{gupta-durrett-2019-entity-tracking,
    title = "Effective Use of Transformer Networks for Entity Tracking",
    author = "Gupta, Aditya  and Durrett, Greg",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
}
```
