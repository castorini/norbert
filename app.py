import time
import sys
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import jnius_config
jnius_config.set_classpath('anserini/target/anserini-0.4.1-SNAPSHOT-fatjar.jar')

from jnius import autoclass
JString = autoclass('java.lang.String')
JSearcher = autoclass('io.anserini.search.SimpleSearcher')

sys.path.append("dl4marco-bert")
import tokenization
from run_msmarco import *


def write_to_tf_record(writer, tokenizer, query, docs):
  query = tokenization.convert_to_unicode(query)
  query_token_ids = tokenization.convert_to_bert_input(
      text=query, max_seq_length=max_query_length, tokenizer=tokenizer, 
      add_cls=True)

  query_token_ids_tf = tf.train.Feature(
      int64_list=tf.train.Int64List(value=query_token_ids))

  for i, doc_text in enumerate(docs):
    doc_token_id = tokenization.convert_to_bert_input(
        text=tokenization.convert_to_unicode(doc_text),
        max_seq_length=max_seq_length - len(query_token_ids),
        tokenizer=tokenizer,
        add_cls=False)

    doc_ids_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=doc_token_id))

    labels_tf = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[0]))

    features = tf.train.Features(feature={
        'query_ids': query_token_ids_tf,
        'doc_ids': doc_ids_tf,
        'label': labels_tf,
    })
    example = tf.train.Example(features=features)
    writer.write(example.SerializeToString())


app = Flask(__name__)
CORS(app)

with app.app_context():
    max_query_length = 64
    max_seq_length = 512

    k1 = 0.82
    b = 0.72

    rm3 = False
    fbTerms = 10
    fbDocs = 10
    originalQueryWeight = 0.5

    searcher = JSearcher(JString("model/lucene-index-msmarco"))
    searcher.setBM25Similarity(float(k1), float(b))
    print('Initializing BM25, setting k1={} and b={}'.format(k1, b))

    if rm3:
        searcher.setRM3Reranker(fbTerms, fbDocs, originalQueryWeight)
        print('Initializing RM3, setting fbTerms={}, fbDocs={} and originalQueryWeight={}'.format(fbTerms, fbDocs, originalQueryWeight))

    tf.logging.set_verbosity(tf.logging.INFO)
    tokenizer = tokenization.FullTokenizer(vocab_file="model/uncased_L-24_H-1024_A-16/vocab.txt", do_lower_case=True)
    bert_config = modeling.BertConfig.from_json_file("model/uncased_L-24_H-1024_A-16/bert_config.json")
    
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint="model/BERT_Large_trained_on_MSMARCO/model.ckpt-100000",
        learning_rate=0,
        num_train_steps=0,
        num_warmup_steps=0,
        use_tpu=False,
        use_one_hot_embeddings=False)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir="output",
        save_checkpoints_steps=None,
        tpu_config=None)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=1,
        eval_batch_size=1,
        predict_batch_size=1)

@app.route("/")
def home():
    return str("Server is up and running.")

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    query = request.args.get("query")
    hits = request.args.get("hits")

    doc_hits = searcher.search(JString(query.encode('utf8')), int(hits))
    docs = [hit.content for hit in doc_hits]

    writer = tf.python_io.TFRecordWriter("query.tf")
    write_to_tf_record(
        writer=writer,
        tokenizer=tokenizer,
        query=query,
        docs=docs
    )
    writer.flush()
    writer.close()
    
    eval_input_fn = input_fn_builder(
        dataset_path="query.tf",
        seq_length=max_seq_length,
        is_training=False,
        max_eval_examples=None)

    predictions = estimator.predict(input_fn=eval_input_fn, yield_single_examples=True)
    results = []
    for item in predictions:
        print(item)
        results.append((item["log_probs"], item["label_ids"]))

    print(len(results))
    log_probs, _ = zip(*results)
    log_probs = np.stack(log_probs).reshape(-1, 2)
    scores = log_probs[:, 1]
    pred_docs = scores.argsort()[::-1]

    ranked_docs = [docs[i] for i in pred_docs]
    return jsonify(ranked_docs)
