FROM python:3.8-slim

RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html -f https://download.pytorch.org/whl/torch_stable.html

CMD ["python", "EssSubgraph.py", "--epochs", "200", "--device", "0", "--dataset", "./data/esssubgraph_human_pc50_string.pkl"]

