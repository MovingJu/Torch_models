FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt install -y libgl1 libglib2.0-0

COPY ./pyproject.toml /app

RUN pip install . --break-system-packages

COPY . /app

RUN python3 modules/Facial_keypoints/path.py

CMD ["python3", "modules/Facial_keypoints/Facial_keypoints.py"]