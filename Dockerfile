# Gunakan image python yang ringan
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Salin requirements dan install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh source code ke container
COPY . .

# Expose port untuk streamlit
EXPOSE 8501

# Jalankan streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
