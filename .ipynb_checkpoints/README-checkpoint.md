# Natural Questions Long Answer Task

This is a simple repo to finetune pretrained [HuggingFace's Transformer](https://huggingface.co/transformers/quickstart.html) 
model on Natural Question (NQ) task.

Natural Questions (NQ) contains real user questions issued to Google search, and
answers found from Wikipedia by annotators. Check out sub-directory [nq](nq) 
for more details.

### Requirement Before Running with CUDA Installed
```
pip install transformers==4.5.0
pip install apache_beam
```

### Download the Dataset
Go to [NQ dataset site](https://ai.google.com/research/NaturalQuestions/download) and download the 
[nq-dev-all set](https://storage.cloud.google.com/natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz) and 
[nq-dev-sample set](https://storage.cloud.google.com/natural_questions/v1.0/sample/nq-dev-sample.jsonl.gz).
Training data is fetched through Transformer's dataset utility.

### Train Model
The whole process includes preprocessing, training and evaluation steps.

```
python3 run_exp.sh
```

### Architectual Decision and Details
The base model is the bert-large-cased-whole-word-masking-finetuned-squad, which is the most suitable pretrained QA model because the vocabulary is similar.
However, the most difficult part of NQ task is its long context size comparing to SQUAD. For the NQ-specific details, I follow some of the preprocess techniques
from [BERT_JOINT](https://arxiv.org/pdf/1901.08634.pdf) such as downsampling negative answers and adding special markup tokens appearing hyperlinks.

To tackle the context size issue, one possible solution is to modify the embeddings type from absolute to **relative position embeddings** as 
proposed in [Improve Transformer Models with Better Relative Position Embeddings](https://arxiv.org/pdf/2009.13658.pdf). This method improves
the F1 on SQUAD for about 2% as it is more robust for long sequence. I tried to use the relative position embeddings and finetined from the same model.
However, it does not perform better than the absolute case. This might be the results of the limitation of single-gpu computation. If **more hardware** is 
available, it could be an approach. 

State of the Art

Ensemble



### F1 Results
The results are tested on validation set of size 7830. (As during validation step during training, we use samples splitted from 
original training set that are not included in the training set)

| F1       | Precision| Recall  |
|----------|----------|---------|
| xx       | 72.9%    | xx      |


