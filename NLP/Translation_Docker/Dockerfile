FROM jupyter/tensorflow-notebook

USER $NB_UID
ENV TRANSFORMERS_CACHE=/hf_cache
RUN pip install --no-cache-dir --upgrade pip
RUN pip install transformers==4.30.2
RUN pip install --no-cache-dir pysrt
RUN fix-permissions "/home/${NB_USER}"