# Local reproduction of the Hugging Face Spaces "streamlit" SDK build
# (README.md: sdk: streamlit, sdk_version: 1.46.0, python_version: "3.12").
# HF Spaces provisions Python 3.12 on a slim Debian base, runs
# `pip install -r requirements.txt`, then launches app_file with streamlit —
# this Dockerfile reproduces that sequence step for step.
FROM python:3.12-slim

WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Mirrors the thread/determinism env app.py sets for the `redstack rank`
# subprocess it shells out to.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
