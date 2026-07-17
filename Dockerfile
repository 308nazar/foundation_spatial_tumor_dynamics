FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3.11-venv && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/project
COPY requirements.txt .
RUN python3.11 -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip && /opt/venv/bin/pip install -r requirements.txt
COPY . .
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/opt/project/src
ENTRYPOINT ["python", "-m", "foundation_spatial_tumor_dynamics.cli.train"]
