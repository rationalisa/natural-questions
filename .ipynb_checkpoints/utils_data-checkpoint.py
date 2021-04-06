import apache_beam as beam
from gzip import GzipFile
from collections import defaultdict
from datasets import Dataset
import json
import six
import re



def _parse_example(line):
    if six.PY2:
        import HTMLParser as html_parser  # pylint:disable=g-import-not-at-top

        html_unescape = html_parser.HTMLParser().unescape
    else:
        import html  # pylint:disable=g-import-not-at-top

    html_unescape = html.unescape
    """Parse a single json line and emit an example dict."""
    ex_json = json.loads(line)
    html_bytes = ex_json["document_html"].encode("utf-8")

    def _parse_short_answer(short_ans):
        """"Extract text of short answer."""
        ans_bytes = html_bytes[short_ans["start_byte"] : short_ans["end_byte"]]
        # Remove non-breaking spaces.
        ans_bytes = ans_bytes.replace(b"\xc2\xa0", b" ")
        text = ans_bytes.decode("utf-8")
        # Remove HTML markup.
        text = re.sub("<([^>]*)>", "", html_unescape(text))
        # Replace \xa0 characters with spaces.
        return {
            "start_token": short_ans["start_token"],
            "end_token": short_ans["end_token"],
            "start_byte": short_ans["start_byte"],
            "end_byte": short_ans["end_byte"],
            "text": text,
        }

    def _parse_annotation(an_json, tokens):
        span = an_json["long_answer"]
        start_tok = span["start_token"]
        end_tok = span["end_token"]
        chars = [len(s) for s in tokens]

        init , end = -1, -1
        if start_tok != -1:
            init = sum(chars[:start_tok])+ start_tok
            end = init + sum(chars[start_tok:end_tok])+ end_tok - start_tok-1
            
        return {
            # Convert to str since some IDs cannot be represented by datasets.Value('int64').
            "id": str(an_json["annotation_id"]),
            "long_answer": {
                "start_token": an_json["long_answer"]["start_token"],
                "end_token": an_json["long_answer"]["end_token"],
                "start_char": init,
                "end_char": end,
                
            },
            "short_answers": [_parse_short_answer(ans) for ans in an_json["short_answers"]],
            "yes_no_answer": an_json["yes_no_answer"] #(-1 if an_json["yes_no_answer"] == "NONE" else an_json["yes_no_answer"]),
        }

    def _clean_token(token):
        """Returns token in which blanks are replaced with underscores.
        HTML table cell openers may contain blanks if they span multiple columns.
        There are also a very few unicode characters that are prepended with blanks.
        Args:
        token: Dictionary representation of token in original NQ format.
        Returns:
        String token.
        """
        return re.sub(u" ", "_", token)
    
    beam.metrics.Metrics.counter("nq", "examples").inc()
    # Convert to str since some IDs cannot be represented by datasets.Value('int64').
    tokens = [_clean_token(t['token']) for t in ex_json['document_tokens']]
    text = " ".join(tokens) #[" ".join(t) for t in tokens]
    id_ = str(ex_json["example_id"])
    return{
            "id": id_,
            "question": ex_json["question_text"],
            "context": text,
            "annotations": [_parse_annotation(an_json, tokens) for an_json in ex_json["annotations"]],

  }
            
'''
"document": {
    "title": ex_json["document_title"],
    "url": ex_json["document_url"],
    "html": ex_json["document_html"],
    "tokens": [
        {"token": t["token"], "is_html": t["html_token"]} for t in ex_json["document_tokens"]
    ],
},
"question": {"text": ex_json["question_text"], "tokens": ex_json["question_tokens"]},
'''

def get_nq_tokens(simplified_nq_example, i):
    '''Returns list of blank separated tokens.'''

    if "context" not in simplified_nq_example:
        raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
                     "example that contains the `document_text` field.")

    return simplified_nq_example["context"][i].split(" ")


def simplify_nq_example(nq_example):
  r"""Returns dictionary with blank separated tokens in `document_text` field.
  Removes byte offsets from annotations, and removes `document_html` and
  `document_tokens` fields. All annotations in the ouput are represented as
  [start_token, end_token) offsets into the blank separated tokens in the
  `document_text` field.
  WARNING: Tokens are separated by a single blank character. Do not split on
    arbitrary whitespace since different implementations have different
    treatments of some unicode characters such as \u180e.
  Args:
    nq_example: Dictionary containing original NQ example fields.
  Returns:
    Dictionary containing `document_text` field, not containing
    `document_tokens` or `document_html`, and with all annotations represented
    as [`start_token`, `end_token`) offsets into the space separated sequence.
  """

  def _clean_token(token):
    """Returns token in which blanks are replaced with underscores.
    HTML table cell openers may contain blanks if they span multiple columns.
    There are also a very few unicode characters that are prepended with blanks.
    Args:
      token: Dictionary representation of token in original NQ format.
    Returns:
      String token.
    """
    return re.sub(u" ", "_", token)

  tokens = [[_clean_token(t) for t in example['tokens']['token']] for example in nq_example['document']]
  text = [" ".join(t) for t in tokens]
  #text = " ".join([_clean_token(t) for t in nq_example["document_tokens"]])

  def _remove_html_byte_add_char_offsets(span, tokens):
    if "start_byte" in span:
      del span["start_byte"]

    if "end_byte" in span:
      del span["end_byte"]

    start_tok = span["start_token"]
    end_tok = span["end_token"]

    chars = [len(s) for s in tokens]


    if not isinstance(start_tok,list):
        init , end = -1, -1
        if start_tok != -1:
            init = sum(chars[:start_tok])+ start_tok
            end = init + sum(chars[start_tok:end_tok])+ end_tok - start_tok-1
        span['start_char']=init
        span['end_char']=end
        '''
        init, end =[],[]
        for i in range(len(start_tok)):
            s = sum(chars[:start_tok[i]])+ start_tok[i]
            init.append(s)
            end.append(s + sum(chars[start_tok[i]:end_tok[i]])+ end_tok[i] - start_tok[i]-1)
        '''



    return span

  def _clean_annotation(annotation, tokens):
    for i,(la, t) in enumerate(zip(annotation, tokens)):
        annotation[i]["long_answer"]=_remove_html_byte_add_char_offsets(la["long_answer"][0], t)
        del annotation[i]["short_answers"]
    '''
    annotation["long_answer"] = [
        _remove_html_byte_add_char_offsets(la["long_answer"][0], t) for (la, t) in zip(annotation, tokens)
    ]
    annotation["short_answers"] = [
        _remove_html_byte_add_char_offsets(sa, tokens) for sa in annotation["short_answers"]
    ]
    '''
    
    return annotation


  simplified_nq_example = {
      "question": [example["text"] for example in nq_example["question"]],
      "id": nq_example["id"],
      #"document_url": [example["url"] for example in nq_example["document"]],
      "context": text,
      "annotations": _clean_annotation(nq_example["annotations"], tokens)
  }

  for i in range(len(nq_example['document'])):
    if len(get_nq_tokens(simplified_nq_example, i)) != len(
        nq_example['document'][i]['tokens']['token']):
      raise ValueError("Incorrect number of tokens.")

  return simplified_nq_example


# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        n_best_size=20,
        output_dir= args.output_dir,
        is_world_process_zero=trainer.is_world_process_zero(),
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    #references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]

    references = [{"id": ex['id'], 'long_answer': {
        'start_token': ex['annotations']['long_answer']['start_token'], 'end_token': ex['annotations']['long_answer']['end_token']
      }} for ex in examples ]

    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def read_annotation_gzip(gzipped_input_file):
  """Read annotation from one split of file."""
  if isinstance(gzipped_input_file, str):
    gzipped_input_file = open(gzipped_input_file, 'rb')
  annotation_dict = defaultdict(list)
  with GzipFile(fileobj=gzipped_input_file) as input_file:
    for line in input_file:
      ex = _parse_example(line)
      for key in ex:
        annotation_dict[key].append(ex[key])

  return annotation_dict


