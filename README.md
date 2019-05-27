# NorBERT

Flask-based demo for [Anserini](https://github.com/castorini/anserini) + [dl4marco-bert](https://github.com/nyu-dl/dl4marco-bert).

![screenshot](https://github.com/castorini/norbert/blob/master/screenshot.png)

## Instructions
1. Copy `BERT_Large_trained_on_MSMARCO/`, `uncased_L-24_H-1024_A-16/` and `lucene-index-msmarco/` into `model/`
2. Install Flask:
```
pip install flask
pip install flask_cors
```
3. Start the app
```
flask run
```
4. Open `index.html`
