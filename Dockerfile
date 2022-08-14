# python-3.9.10, jupyter-lab starck git commit:8be708bd9c07
FROM jupyter/scipy-notebook@sha256:6889c7df60b393a25a73ed0b18fe421ce173f1a01fe3b723793a108fa18cd8b5

# Install in the default python3 environment
RUN pip install --quiet --no-cache-dir \
    'qiskit[visualization]==0.34.0' \
    hydra-core==1.1.2 \
    xgboost==1.6.0 \
    tensorboard==2.8.0 \
    tensorboardX==2.5 \
    crc32c==2.2.post0 && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
