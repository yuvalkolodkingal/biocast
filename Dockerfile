# Container for the design studio. Serves two targets from one image:
#   Hugging Face Spaces  — sdk: docker, expects the app on port 7860
#   Cloud Run            — injects $PORT (8080); needs --session-affinity at deploy time
# Every dependency ships as a wheel on linux/amd64 and linux/aarch64, so there is no
# build toolchain and no apt package to install.
FROM python:3.13-slim

# Spaces runs the container as uid 1000 and will not let root write the app directory,
# so create that user up front and install as them.
RUN useradd --create-home --uid 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /home/user/app
USER user

COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . ./

# 7860 is the Spaces default; Cloud Run overrides PORT at runtime.
ENV PORT=7860
EXPOSE 7860

# enableCORS=false is required to run behind the Spaces iframe proxy. XSRF protection
# is left ON — the app takes no uploads, so nothing needs it relaxed.
CMD ["sh", "-c", "streamlit run streamlit_app.py \
     --server.port=${PORT:-7860} \
     --server.address=0.0.0.0 \
     --server.headless=true \
     --server.enableCORS=false"]
