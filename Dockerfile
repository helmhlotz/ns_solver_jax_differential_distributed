FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_FLAGS=--xla_force_host_platform_device_count=8

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

COPY . /app

RUN mkdir -p /app/result

CMD ["python", "runner.py", "--config", "sample_config/config_N_128.json"]
