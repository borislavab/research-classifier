# Research Classifier

This project is a web application for classifying research papers into multiple categories based on their abstracts.  
The repo includes fine-tuning a pre-trained BERT bert-base-cased model based on the [ArXiv dataset from Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).  
The problem is a multi-label classification problem with 176 original categories, 2635119 samples and significant class imbalance.  
For the training I used Google Colab with a T4 GPU which connects to Google Drive for storage.  
The notebook can be found in the [research_classifier.ipynb](research_classifier/training/research_classifier.ipynb) file.

# Dataset analysis and decision records

Refer to the [analysis.ipynb](research_classifier/analysis/analysis.ipynb) notebook for dataset analysis conducted, helpful visualizations, alternative approaches considered and decision chosen.

# How to run locally

1. Create and activate a conda environment:

```bash
conda create -n .conda python=3.11
conda activate .conda
```

2. Install dependencies into the conda environment:

```bash
conda install --file requirements.txt
pip install -e .
```

Install additional dependencies for the web layer (celery, redis, etc.) if using the API or management commands:

```bash
pip install -r requirements/requirements_web.txt
```

# Management commands

I added a few utility commands to manage the dataset and train the model:

## Undersample dataset:

```bash
python manage.py undersample_dataset path/to/full/dataset output/path/for/undersampled/dataset --threshold optional_threshold
```

## Run training:

```bash
python manage.py train_model --output-dir path/to/output --dataset path/to/full/dataset --num-epochs 3
```

If path to dataset is not provided, it will be downloaded from KaggleHub.  
By default this will resume from the a checkpoint it finds in the output directory.  
Pass `--from-scratch` to start training from scratch.  
`--sample-count` parameter is also exposed for testing and for decreasing the dataset size. If not specified, the full dataset is used.

## Run evaluation:

```bash
python manage.py evaluate_model --model-path path/to/checkpoint --output-dir path/to/output --dataset path/to/full/dataset
```

`sample-count` parameter is also exposed for testing and for decreasing the dataset size. If not specified, the full dataset is used.

## Run oversampling:

This uses a locally running the Llama 3.2 3B model. To run it install ollama on your machine and run:

```bash
ollama run llama3.2
```

To generate `target_count` samples for a given set of labels run:

```bash
python manage.py oversample_dataset path/to/full/dataset path/to/output/file.jsonl target_count label1 label2 ...
```

# Run the web app:

The app looks for a model checkpoint in the `research_classifier/prediction/model` directory.
TODO: Update with instructions to download the model checkpoint.

Since the app uses Celery with a Redis backend, start a Redis container:

```bash
docker run -d --name redis-server -p 6379:6379 redis:latest
```

Advice from personal experience: Don't forget to stop the container later!! :)

(Optional) To check connectivity to Redis (on Mac):

```bash
brew install redis
redis-cli ping # should return PONG
```

Celery workers will load a model checkpoint to run predictions.
They expect the checkpoint to be in the `research_classifier/prediction/model` directory by default,  
but if you downloaded in a different path, you can supply it by setting the MODEL_CHECKPOINT_PATH environment variable:

```bash
export MODEL_CHECKPOINT_PATH=/path/to/checkpoint
```

Now from the top level directory, start the Celery worker:

```bash
celery -A research_classifier worker --loglevel=info
```

Run the server:

```bash
python manage.py runserver
```

Make a POST request to submit a prediction request to the server:

```bash
curl http://127.0.0.1:8000/api/predict/ -H "Content-Type: application/json" -d '{"article": "This is a test article about machine learning."}'
```

Sample response:

```json
{
  "task_id": "85df8632-8fa5-4692-9a23-b8e508405b0d",
  "status": "pending",
  "created_at": "2025-01-07T23:33:58.086598"
}
```

This is an asynchronous API for a long running operation, so the client is expected to poll for the results.  
For this the HTTP payload returns standard fields:

> HTTP/1.1 202 Accepted
> Retry-After: 1
> Location: /api/prediction/b35e5c2c-3674-46da-aa6c-0d2c272b94a7

Following the location link with a GET request will return the results of the operation:

```bash
curl http://127.0.0.1:8000/api/prediction/<task_id>
```

Sample response:

```json
{ "predictions": ["cs.LG", "stat.ML"], "status": "success" }
```

with status 200.
While processing it returns a 202 status with a Retry-After header with pending/processing status.
On an error it returns an error message with an appropriate error status.  
The task result is stored in Redis for 10 minutes, after which it expires and cannot be retrieved.

> **Note:** The API is asynchronous since it's expected that model prediction is a CPU-bound task
> which is better to be ran in a separate process from the web server to ensure responsiveness and availability.  
> Celery could easily scale workers count and workers can be distributed across multiple servers,
> so this way the app can scale horizontally with request load.  
> Alternative to polling for responses, Django channels and websockets could be used to stream responses to the client.

# Run tests

From the top level directory:

```bash
pytest
```

# Generate requirements files

For different purposes different requirements files are generated - i.e. Google Colab uses pip package manager. I'm using a conda environment for development.

From an activated conda environment:

- conda list -e > requirements.txt
- conda env export > environment.yml
- pip list --format=freeze > requirements_pip.txt

For ease of development I'm using one conda environment for both the web app and training the model. For production use cases separate environments should be used so the web server wouldn't have to install the model training dependencies - example file structure in [requirements](requirements) folder.

# Improvements proposed

- [ ] Run the training with a dataset containing oversampled examples from the least performing classes as identified in the analysis notebook
- [ ] Implement more sophisticated undersampling for multi-label classification - https://www.din.uem.br/yandre/Neurocomputing_MLTL.pdf
- [ ] Compute custom loss like focal loss or weighted cross entropy for the class imbalance problem
- [ ] Implement processing of formulas and scientific notation in the abstracts
- [ ] Serialize and deserialize the pipeline settings to be stored along with the saved model for inference
- [ ] Train a custom tokenizer with extended vocabulary for scientific terms
