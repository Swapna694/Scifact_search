# SciFact Search

Command-line search over the SciFact dataset using:
- Keyword search: TF-IDF and BM25 (implemented from scratch)
- Semantic search: sentence embeddings + cosine similarity

## Setup

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


## Run

python main.py                    # interactive search
python run_queries.py --n 20      # compare all methods on test queries

The dataset downloads automatically on the first run.
