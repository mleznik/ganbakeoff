FROM pytorch/pytorch


# VARIABLES--------------------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive


# COPY FILES TO IMAGE----------------------------------------------------------------------------
RUN mkdir -p /opt1/out/
RUN mkdir -p /opt1/program/
RUN mkdir -p /opt1/config/
RUN mkdir -p /opt1/out/wandb/
RUN mkdir -p /opt1/data/
COPY requirements.txt /opt1/config
COPY wandbkey.json  /opt1/config


# INSTALL BACKEND DEPENDENCIES-------------------------------------------------------------------
RUN apt-get update -y

RUN apt install build-essential -y


# INSTALL PYTHON PACKAGES------------------------------------------------------------------------
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /opt1/config/requirements.txt


# DEFINE ENTRYPOINT------------------------------------------------------------------------------
ENTRYPOINT ["/bin/bash", "/opt1/program/run.sh"]

