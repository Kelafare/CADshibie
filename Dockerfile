# 离线评测镜像：CPU-only，不含任何外部模型 API 依赖。
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app/submission

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 构建阶段做一次模型清单/依赖自检；失败则不产出镜像。
RUN python tools/check_compliance.py --out /tmp/compliance_report.json

# 评测平台通常在容器内执行等价命令：
#   python infer.py --input /data/dev_final --output /results --weights /app/submission/weights
# 默认入口只调用本地推理，不访问网络。
ENTRYPOINT ["python", "infer.py"]
CMD ["--input", "/data/dev_final", "--output", "/results", "--weights", "/app/submission/weights", "--workers", "4"]
