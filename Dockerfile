# Python 3.11 to match training; "slim" keeps the base image small
FROM public.ecr.aws/docker/library/python:3.11-slim

# Lambda Web Adapter: lets this ordinary web server run unchanged on AWS Lambda
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app

# Pinned libraries first, so this slow layer is reused when only the code changes
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# The int8 classifier, the outcome model and its similar-applications index, then the code
COPY models/category-onnx/ models/category-onnx/
COPY models/outcome.json models/
COPY models/similar.npz models/
COPY src/planscope/ planscope/

ENV MODELS_DIR=/app/models \
    AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health

CMD ["uvicorn", "planscope.api:app", "--host", "0.0.0.0", "--port", "8000"]