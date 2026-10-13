FROM python:3.12-slim
RUN pip install --no-cache-dir python-docx==1.2.0
COPY app/infrastructure/report_export_worker.py /worker.py
USER 65534:65534
ENTRYPOINT ["python", "-I", "/worker.py"]
