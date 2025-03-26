FROM python:3.9.19 as compiler

RUN apt-get update

WORKDIR /client

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN python3 -m pip install --upgrade pip

RUN apt-get update && apt-get install -y \
  gcc \
  gfortran \
  g++ \
  build-essential

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m pip install --upgrade pip
RUN apt-get install -y proj-bin
ENV PROJ_DIR=/usr

RUN pip install pyproj --no-cache-dir

RUN pip install pygrib
RUN pip install --upgrade pip

RUN pip install cfgrib --no-cache-dir
RUN pip install scikit-fuzzy --no-cache-dir

RUN pip install h5py joblib tqdm lmdb --no-cache-dir

RUN pip install pvlib --no-cache-dir
RUN pip install workalendar credentials --no-cache-dir

RUN pip install yagmail --no-cache-dir
RUN pip install xgboost --no-cache-dir
RUN pip install threadpoolctl --no-cache-dir
RUN pip install xlrd==1.2.0 --no-cache-dir
RUN pip install earthengine-api --no-cache-dir
RUN pip install mysql-connector-python --no-cache-dir
RUN pip install wget --no-cache-dir

RUN apt-get update \
    && apt-get install -y  g++ binutils libproj-dev gdal-bin  --no-install-recommends && \
    apt-get clean -y

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN pip install rasterio --no-cache-dir
RUN pip install openpyxl --no-cache-dir
#RUN cd /root
#RUN mkdir -p .config
#RUN cd .config
#RUN mkdir -p earthengine
#RUN cd earthengine
#COPY credentials /root/.config/earthengine/
RUN cd /
RUN mkdir -p /nwp/
RUN mkdir -p /predictions/

RUN mkdir -p /models/
RUN pip install pymysql --no-cache-dir
RUN pip install sqlalchemy --no-cache-dir
RUN pip install opencv-python xarray xlsxwriter --no-cache-dir
RUN pip install scikit-image astral --no-cache-dir
RUN pip install pyodbc --no-cache-dir

RUN pip install catboost==1.2.2 --no-cache-dir

RUN pip install scikit-learn==1.4.2 --no-cache-dir

RUN pip install numpy==1.25.2 pandas==2.1.1
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
RUN pip install gpytorch --no-cache-dir
RUN pip install reformer-pytorch
RUN pip install eumdac
RUN pip install psutil gpu-utils
RUN pip install transformers timm einops
RUN pip install dask

WORKDIR /client
COPY . .

CMD ["python", "run_model_online.py"]

