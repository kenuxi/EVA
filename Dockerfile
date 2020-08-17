FROM python:3.7-slim
RUN pip install -r requirements.txt
WORKDIR /eva
ADD . /eva
CMD ["python", "app.py"]
