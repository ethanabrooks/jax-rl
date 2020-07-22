FROM nvcr.io/nvidia/pytorch:19.10-py3

ENV PYTHON_VERSION=cp36
ENV CUDA_VERSION=cuda101
ENV BASE_URL='https://storage.googleapis.com/jax-releases'
ENV PLATFORM=linux_x86_64
RUN pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.48-$PYTHON_VERSION-none-$PLATFORM.whl

COPY ./environment.yml /tmp/environment.yml
RUN conda env update -f /tmp/environment.yml \
    && conda clean --all -y

RUN apt-get update && apt-get install -y rsync  && rm -rf /var/lib/apt/lists/*

RUN echo "source activate base" >> /root/.bashrc
ENV PATH /opt/conda/envs/jax/bin:$PATH

COPY entrypoint.sh /
WORKDIR /jax
COPY . "/jax"
ENTRYPOINT ["/entrypoint.sh"]
