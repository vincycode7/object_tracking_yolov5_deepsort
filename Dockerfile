FROM python:3.10.6
ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update -y
RUN apt-get install -y build-essential
RUN apt-get install -y python3-pip
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
EXPOSE 8501
COPY . /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENTRYPOINT [ "streamlit","run" ]
CMD ["Home.py"]