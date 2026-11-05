# Image for train.py, train_privatefl.py and pretrain.py; the repository is mounted at /workspace.
# Build: docker build -t subfl .
# Run from the repository root:
#   docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/workspace subfl bash run.sh experiments/main.txt
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace
ENV PYTHONPATH=/workspace MPLBACKEND=Agg PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir numpy==2.2.5 pandas==3.0.5 scipy==1.15.3 matplotlib==3.10.8

CMD ["bash"]
