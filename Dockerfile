FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
USER root

WORKDIR /app
COPY requirements.txt .

# 安装依赖包
RUN pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

COPY server.py .

ENV API_AUTH_KEY=mt_photos_ai_extra
ENV RECOGNITION_MODEL=buffalo_l
ENV DETECTION_THRESH=0.65
EXPOSE 8066

VOLUME ["/root/.insightface/models"]

CMD [ "python3", "server.py" ]
