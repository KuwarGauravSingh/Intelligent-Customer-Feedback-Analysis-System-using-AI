FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy repo content
COPY . /app

# Download spaCy model
RUN python -m spacy download en_core_web_sm

EXPOSE 8501
CMD ["streamlit", "run", "src/app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
