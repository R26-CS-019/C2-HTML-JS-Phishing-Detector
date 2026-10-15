# C2 HTML/JS phishing detector API for Hugging Face Spaces (Docker SDK).
# Builds from the GitHub repo, so the same file works when pasted into a Space.
FROM python:3.12.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONDONTWRITEBYTECODE=1
WORKDIR /home/user

ARG REPO=https://github.com/R26-CS-019/C2-HTML-JS-Phishing-Detector.git
ARG REF=main
RUN git clone --depth 1 --branch "$REF" "$REPO" app
WORKDIR /home/user/app

RUN pip install --no-cache-dir --user -r requirements_api.txt

# Fail the build early if the frozen model check does not pass
RUN python 29_run_webpage_detector.py --check

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
