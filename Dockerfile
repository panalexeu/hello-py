FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages upfront
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn

WORKDIR /workspace

# Copy files
COPY ./data/new_train.csv ./train.csv
COPY ./data/test.csv .
COPY ./data/sample_submission.csv .