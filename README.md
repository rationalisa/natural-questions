# Natural Questions

This is a simple repo to finetune pretrained [HuggingFace's Transformer](https://huggingface.co/transformers/quickstart.html) 
model on Natural Question (NQ) task.

Natural Questions (NQ) contains real user questions issued to Google search, and
answers found from Wikipedia by annotators. Check out sub-directory [nq](nq) 
for more details.

# Requirement Before Running with CUDA Installed
```
pip install transformers==4.5.0
pip install apache_beam
```

# Download the Dataset
Go to [NQ dataset site](https://ai.google.com/research/NaturalQuestions/download) and download the 
[nq-dev-all set](https://storage.cloud.google.com/natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz) and 
[nq-dev-sample set](https://storage.cloud.google.com/natural_questions/v1.0/sample/nq-dev-sample.jsonl.gz).
Training data is fetched through Transformer's dataset utility.

# Train Model
The whole process includes preprocessing, training and evaluation steps. All the steps are included in `run_exp.sh`.
Modify `root_dir` and `dir_name` before training.
```
python3 run_exp.sh
```




### F1 Results
The results are tested on validation set of size 7830. (As during validation step during training, we use samples splitted from 
original training set that are not included in the training set)

| F1       | Precision| Recall  |
|----------|----------|---------|
| xx       | 72.9%    | xx      |


