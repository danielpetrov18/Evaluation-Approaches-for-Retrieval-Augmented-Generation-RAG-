FROM python:3.12.10-slim

WORKDIR /frontend

# Copy only requirements.txt first to leverage Docker's caching (Check out image layering)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# streamlit run st_app.py
CMD ["streamlit", "run", "st_app.py"]