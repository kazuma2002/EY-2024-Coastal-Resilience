FROM python:3.8

# 環境変数の設定
ENV PYTHONUNBUFFERED 1

# 作業ディレクトリの設定
WORKDIR /workspace

# 依存関係のインストール
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt
