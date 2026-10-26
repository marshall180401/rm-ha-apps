FROM python:3.12-alpine
RUN apk add --no-cache ffmpeg
RUN pip install --no-cache-dir aiohttp==3.10.10
WORKDIR /app
COPY bridge.py .
CMD ["python", "-u", "bridge.py"]
