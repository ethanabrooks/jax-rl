FROM nvcr.io/nvidia/pytorch:19.10-py3
RUN pip install --upgrade pip gym
RUN conda remove wrapt
RUN pip install \
        "tensorboardX==1.8" \
        "opencv-python==4.1.0.25" \
        "requests==2.22.0" \
        "black==19.10b0" \
        "dm-env==1.2" \
        "dm-haiku==0.0.1" \
        "flax==0.1.0rc2" \
        "ipdb==0.13.2" \
        "numpy==1.18.0" \
        "ipython==7.15.0" \
        "tensorflow==2.2.0" \
        "tensorboard==2.2.2"

ENV PYTHON_VERSION=cp36
ENV CUDA_VERSION=cuda101
ENV BASE_URL='https://storage.googleapis.com/jax-releases'
ENV PLATFORM=linux_x86_64
RUN pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.48-$PYTHON_VERSION-none-$PLATFORM.whl
RUN pip install --upgrade jax

COPY entrypoint.sh /
WORKDIR /sac
COPY . "/sac"
ENTRYPOINT ["/entrypoint.sh"]
