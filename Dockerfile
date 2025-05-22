FROM python:3.12-slim

# 작업 디렉토리
WORKDIR /app

# 종속성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사
COPY . /app

# 서비스 기동
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8007"]
