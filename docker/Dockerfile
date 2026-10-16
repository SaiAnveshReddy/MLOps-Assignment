FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal on purpose - scikit-learn/pandas wheels are
# manylinux, so no compiler toolchain is needed here.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
RUN pip install --no-cache-dir -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
