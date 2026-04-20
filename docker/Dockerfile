FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    jupyter==1.1.1 \
    nest-asyncio==1.6.0 \
    polars==1.39.3 \
    pandas==3.0.2 \
    pyarrow==23.0.1 \
    numpy==2.4.4 \
    scipy==1.17.1 \
    scikit-learn==1.6.1 \
    pingouin==0.6.1 \
    statsmodels==0.14.4 \
    plotly==6.6.0 \
    seaborn==0.13.2 \
    matplotlib==3.10.8 \
    kaleido==1.2.0 \
    tqdm==4.67.3 \
    emoji==2.14.1 \
    "pydantic==2.12.3" \
    requests==2.32.3

WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
