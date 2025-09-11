# CUDA 11.4 + cuDNN runtime (pairs safely with a driver that reports CUDA 11.4)
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

# ---------- Build-time proxy args (do NOT persist) ----------
# These are available during `RUN` steps but are NOT kept in the final image.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy

# Base deps + tini for signal handling
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash ca-certificates curl bzip2 git tini locales && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Create a non-root user
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME}

# Micromamba (drop-in conda replacement)
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# Make micromamba available in every shell and auto-activate last-used env if you want
RUN echo 'eval "$(micromamba shell hook -s bash)"' >> /etc/bash.bashrc

# Work in your repo
WORKDIR /workspace
# (We’ll mount the repo at runtime; copying is optional.)
# COPY . .

# Persist conda envs/caches via a named volume
VOLUME ["/opt/conda"]

# Proper init + non-root
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
USER ${USERNAME}
