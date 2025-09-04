FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt install -y libgl1 libglib2.0-0

COPY ./pyproject.toml /app

RUN pip install . --break-system-packages

COPY ./modules/path.py /app/modules/path.py
RUN python3 modules/path.py Face_detection


COPY ./models /app/models
COPY ./modules /app/modules
COPY ./main.py /app

CMD ["python3", "main.py"]