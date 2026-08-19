# Falcon 9 First Stage Landing Analysis

This project explores historical Falcon 9 launches and estimates the probability that a first
stage lands successfully. It includes a tested Python package, a Dash dashboard, an interactive
inference form, and the original course notebooks retained as learning artifacts.

The prediction is educational. It is trained on a small historical course dataset and is not an
operational SpaceX model.

## What the application provides

1. Landing success rates by launch site
2. Payload, orbit, and outcome exploration
3. A real landing probability produced by the fitted classification pipeline
4. Evaluation metrics calculated on the latest 20 percent of launches
5. Automatic dataset validation and local caching

## Methodology

The model is a logistic regression classifier with preprocessing contained in one scikit learn
pipeline. Numeric values are imputed and standardized. Boolean values are imputed, and categorical
values are one hot encoded with support for unseen categories.

Evaluation uses an ordered temporal split. Earlier launches train the evaluation model and later
launches form an untouched holdout set. After metrics are recorded, the dashboard model is fitted
on all available records for interactive inference. This avoids fitting preprocessing on holdout
data and avoids choosing a winning model from test results.

Accuracy alone can be misleading on a small dataset. The dashboard therefore displays accuracy,
majority class baseline accuracy, balanced accuracy, ROC AUC, F1 score, and holdout size. Precision
and recall are also available through `ModelReport`. The current temporal result does not beat the
simple accuracy baseline, which is reported openly instead of being presented as a production
quality result.

## Run locally

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest
spacex-dashboard
```

Open `http://127.0.0.1:8051`.

The first run downloads the documented IBM course dataset and stores it at
`data/dataset_part_2.csv`. To use an existing file, set `SPACEX_DATA_CACHE` to its path before
starting the application. The download is checked against the documented SHA 256 digest so an
upstream data change cannot silently alter the reported results.

## Deployment

Install the deployment extra and serve the exported WSGI application:

```bash
python -m pip install -e ".[deploy]"
gunicorn app:server
```

The application honors the `PORT` environment variable when started with `spacex-dashboard`.

## Tests and automation

The tests cover schema validation, local data loading, temporal evaluation boundaries, model
inference, and dashboard construction. GitHub Actions runs Ruff and pytest on Python 3.10 and
Python 3.12 for pushes and pull requests.

## Repository map

```text
app.py                         WSGI entry point
src/spacex_falcon/data.py      Data retrieval and validation
src/spacex_falcon/model.py     Preprocessing, evaluation, and inference
src/spacex_falcon/dashboard.py Dash user interface and callbacks
tests/                         Automated tests
docs/PROVENANCE.md             Data and notebook attribution
*.ipynb                        Historical IBM Skills Network exercises
```

## Data and attribution

The processed launch data and the historical notebooks come from the IBM Skills Network Data
Science Capstone course. See `docs/PROVENANCE.md` for scope and attribution details.

No repository license has been selected. The owner should confirm applicable course terms before
licensing or redistributing the historical notebooks.
