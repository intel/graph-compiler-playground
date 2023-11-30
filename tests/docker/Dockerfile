FROM summerwind/actions-runner:ubuntu-22.04

RUN curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update --yes \
    && apt-get install build-essential intel-basekit cmake --yes \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA=/opt/conda
ENV PATH="${CONDA}/bin:${PATH}"

RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh \
    && bash /tmp/miniconda3.sh -b -p "${CONDA}" -f -u \
    && "${CONDA}/bin/conda" init bash \
    && rm -f /tmp/miniconda3.sh
