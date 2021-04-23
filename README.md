# Natural Questions Long Answer Task

This is a simple repo to finetune the pretrained [HuggingFace's Transformer](https://huggingface.co/transformers/quickstart.html) 
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

### Train Model and Evaluate
The whole process includes preprocessing, training and evaluation steps.

```
python3 run_exp.sh
```

### Architectural Decision and Insights
The base model is the bert-large-cased-whole-word-masking-finetuned-squad, which is the most suitable pretrained QA model because the vocabulary is similar, as well as
the long max-sequence-length. However, the most difficult part of the NQ task is its long context size compared to SQUAD. For the NQ-specific details, I followed some
of the preprocess techniquesfrom [BERT_JOINT](https://arxiv.org/pdf/1901.08634.pdf) such as downsampling negative answers and adding special markup tokens appearing 
in hyperlinks. During the post-processing step, I ensembled two models with lowest validation loss and the ensemble model improves the F1 score for 1 percent.

To tackle the context size issue, one possible solution is to modify the embeddings type from absolute to **relative position embeddings** as 
proposed in [Improve Transformer Models with Better Relative Position Embeddings](https://arxiv.org/pdf/2009.13658.pdf). This method improves
the F1 on SQUAD for about 2% as it is more robust for long sequences. I tried to use the relative position embeddings and finetined from the same model.
However, it does not perform better than the absolute case. This might be the result of the limitation of single-gpu computation. If **more hardware** is 
available, it could be an approach. 


Current State of the Art models such as [Cluster-former](https://arxiv.org/pdf/2009.06097.pdf) improve the baseline score significantly. If more time and resources
are available, **sparse transformers** that perform attention hierarchically definitely would boost the performance on long sequences. I think this approach is straightforward 
yet pretty clever for question answering tasks of long sequences. Modification on transformers such as **Reformer** is a future direction. 


### F1 Results
The scores are tested on validation set of size 7830. (As during validation step during training, I use samples split from 
original training set that are not included in the training set)

|   F1    | Precision |  Recall |
|---------|-----------|---------|
|  59.84  |  60.44%   |  59.24% |

### Fav Charity ###
[PETSMART Charities](https://secure.petsmartcharities.org/give/219478/#!/donation/checkout?c_src=pci_web&c_src2=makeadonation_redirect)


