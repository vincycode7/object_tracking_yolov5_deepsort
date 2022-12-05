FROM python:3.10.6
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /app
ENTRYPOINT [ "streamlit","run" ]
CMD ["Home.py"]