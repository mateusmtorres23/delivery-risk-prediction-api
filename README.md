# Olist Delivery Risk

**Delivery risk scoring at order approval for Brazilian e-commerce.**

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![AWS Lambda](https://img.shields.io/badge/AWS%20Lambda-FF9900?style=flat-square)

This ML Zoomcamp midterm project estimates the probability that an order will arrive after its promised delivery date. An operations team could use the score to prioritize a limited review queue while there is still time to act.

[Try the API](#try-the-api) · [Technologies](#technologies) · [Data and modeling](#data-and-modeling) · [Holdout results](#holdout-results) · [Reproduce the project](#reproduce-the-project) · [Deployment](#deployment)

## Try the API

Send `POST /predict` to the live FastAPI service on AWS Lambda. The example describes a single-item order bound for São Paulo with a 10-day promised delivery window. Weight, volume, and seller distance can be omitted when unavailable.

```powershell
$body = @{
    days_from_approval_to_estimate = 10
    item_count = 1
    distinct_product_count = 1
    distinct_seller_count = 1
    price_sum = 100
    freight_sum = 20
    customer_state = 'SP'
} | ConvertTo-Json -Compress

Invoke-RestMethod -Method Post `
  -Uri 'https://cxo3ff3m3qrays2ynkiit7u2wu0fscxh.lambda-url.us-east-2.on.aws/predict' `
  -ContentType 'application/json' `
  -Body $body
```

The sample order returns this probability.

```json
{"late_probability": 0.08744106115954041}
```

### Run locally

The repository includes the trained pickle. You can start the API directly from a clone.

#### Docker

With Docker running, build and start the image from the repository root.

```bash
docker build -t delivery-risk .
docker run --rm -p 127.0.0.1:8000:8000 delivery-risk
```

#### Python

```bash
uv run --locked python -m uvicorn app.app:app --host 127.0.0.1 --port 8000
```

Use the same request body with `http://127.0.0.1:8000/predict`, or try it in the [local API docs](http://127.0.0.1:8000/docs).

## Technologies

| Technology | Role in the project |
| --- | --- |
| Python 3.12 and uv | Run the project with dependencies recorded in `uv.lock`. |
| pandas and NumPy | Clean and join the Olist tables, aggregate order features, and calculate geographic distance. |
| Jupyter notebooks | Explore the dataset, compare models, and report validation and holdout results. |
| scikit-learn | Fit preprocessing and the classifier together in one pipeline. |
| FastAPI, Pydantic, and Uvicorn | Validate order features and serve predictions over HTTP. |
| AWS Lambda and Lambda Web Adapter | Run the FastAPI app behind a Lambda Function URL in US East (Ohio). |

## Data and modeling

The pipeline joins the orders, customers, items, sellers, products, and geolocation tables to produce **96,394 unique delivered orders**. It aggregates item data to the order level and uses median coordinates for each ZIP prefix to estimate seller-to-customer distance. An order is labeled late when its **actual delivery date falls after its promised date**. [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The model uses the promise window, customer state, item, product, and seller counts, price and freight totals, package weight and volume, and the longest seller-to-customer distance in the order. The fitted pipeline imputes missing numeric values with training medians, scales numeric features, and one-hot encodes customer state.

The modeling notebook compares **14 configurations** across logistic regression, decision trees, random forests, and histogram gradient boosting. It tests both the promise window alone and the full order feature set against a constant baseline.

| Stage | Orders used |
| --- | --- |
| Initial fit | Orders approved and delivered before March 1, 2018. |
| Validation | March and April approvals with delivery outcomes recorded before June 1. May gives those outcomes time to arrive. |
| Final fit | **75,388** orders approved and delivered before June 1. |
| Holdout | **18,610** June–August approvals with recorded deliveries. |

The selected model is **full-order logistic regression with `C=0.1`**. Its mean validation average precision was close to the leading boosted model’s (**0.24927 vs. 0.24960**). At the simulated 10% monthly review budget, it found **425 late orders** across March and April compared with **416** for boosting. [Modeling notebook](notebooks/03_modeling.ipynb)

## Holdout results

The holdout late rate was **3.60%**. Reviewing the highest-risk 10% of orders each month caught **131 of 670** late deliveries. Random selection at the same workload would have caught about **67** in expectation.

| Holdout metric | Result |
| --- | ---: |
| Average precision | 0.0799 |
| ROC-AUC | 0.7020 |
| Precision at a 10% monthly review budget | 7.0% |
| Recall at a 10% monthly review budget | 19.6% |

| Month | Orders reviewed | Late orders caught | Recall |
| --- | ---: | ---: | ---: |
| June | 610 | 25 of 71 | 35.2% |
| July | 603 | 47 of 193 | 24.4% |
| August | 650 | 59 of 406 | 14.5% |

These results apply to orders with recorded deliveries and a retrospective queue that ranks each month’s full set of scores.

## Reproduce the project

Use Python 3.12 and uv. Run the commands below from the repository root. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

1. Download the Olist dataset and extract all nine CSVs directly into `data/raw/`. For example, `olist_orders_dataset.csv` goes in `data/raw/olist_orders_dataset.csv`. [Olist dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Install the locked dependencies, prepare the data, and refit the model.

   ```bash
   uv sync --locked
   uv run --locked python pipelines/select_data.py
   uv run --locked python pipelines/clean_data.py
   uv run --locked python train.py
   ```

   The pipeline writes selected CSVs to `data/selected/`, builds `data/processed/logistic_abt.csv`, and regenerates `app/model/delivery_risk.pkl` with the fitted preprocessing and classifier.

3. Launch JupyterLab with `uv run --locked --with jupyterlab jupyter lab` and open the notebooks from `notebooks/`.

   | Notebook | What it covers |
   | --- | --- |
   | [01 · Data audit](notebooks/01_data_audit.ipynb) | Source tables and column selection. |
   | [02 · Data cleaning](notebooks/02_data_cleaning.ipynb) | Cleaning decisions and the order-level joins. |
   | [03 · Modeling](notebooks/03_modeling.ipynb) | EDA, model selection, cutoff experiments, and holdout results. |

## Deployment

The demo uses a **Lambda Function URL → Lambda Web Adapter → FastAPI** in US East (Ohio). The deployment ZIP contains the application and trained pipeline. FastAPI loads the pickle at startup and serves `POST /predict`.

### Build the Lambda ZIP

Run this from the repository root in Linux or WSL with uv on your `PATH`.

```bash
uv python install 3.12
uv run --no-project --python 3.12 python scripts/build_lambda_zip.py
```

The build script stages Linux x86_64 wheels from `uv.lock` and omits the scientific libraries’ bundled test files. It writes `build/delivery-risk.zip` and reports how much space remains for layers under Lambda’s 250 MiB unpacked limit. The ZIP used for the demo exceeds the Lambda console’s 50 MiB direct-upload limit, so the deployment uses S3 for the upload. [Build script](scripts/build_lambda_zip.py) · [AWS ZIP packaging guide](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html)

### Configure Lambda

1. Upload `build/delivery-risk.zip` to a private S3 bucket in `us-east-2`. Create a Lambda **Event Function** with the settings below. In the function’s **Code** tab, update it from the S3 location using **Copy mode**.

   | Setting | Value |
   | --- | --- |
   | Region | US East (Ohio), `us-east-2` |
   | Runtime and architecture | Python 3.12, `x86_64` |
   | Memory and timeout | 512 MB, 30 seconds |
   | Handler | `run.sh` |
   | Web Adapter layer ARN | `arn:aws:lambda:us-east-2:753240598075:layer:LambdaAdapterLayerX86:28` |
   | Environment variables | `AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap`, `PORT=8000` |

2. Attach the Web Adapter layer and apply the handler and environment settings. The ZIP and layers together must fit under the unpacked size limit. [AWS FastAPI ZIP example](https://github.com/aws/aws-lambda-web-adapter/tree/main/examples/fastapi-zip)
3. Enable a **Function URL** with auth type `NONE` for the public demo. Run the request in [Try the API](#try-the-api) with your new Function URL and check the response. [AWS Function URL authentication](https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html)
