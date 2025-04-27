FROM python:3.11.12-slim-buster
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
ENTRYPOINT ["streamlit", "run", "webview.py", "--server.address", "0.0.0.0", "--server.port", "80"]
