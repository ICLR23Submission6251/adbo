FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt /workspace
RUN python -m pip install -r requirements.txt
COPY . /workspace

CMD python -u cp-main-async.py
