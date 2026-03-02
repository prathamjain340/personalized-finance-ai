FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 9001

CMD ["uvicorn", "finance_project.main:app", "--host", "0.0.0.0", "--port", "9001"]
