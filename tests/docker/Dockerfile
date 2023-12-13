FROM summerwind/actions-runner:ubuntu-22.04

RUN curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        | sudo tee /etc/apt/sources.list.d/oneAPI.list \
    && sudo apt-get update --yes \
    && sudo apt-get install build-essential intel-basekit cmake numactl --yes \
    && sudo rm -rf /var/lib/apt/lists/*

ENV CONDA=/opt/conda
ENV PATH="${CONDA}/bin:${PATH}"

RUN sudo curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda3.sh \
    && sudo bash /tmp/miniconda3.sh -b -p "${CONDA}" -f -u \
    && "${CONDA}/bin/conda" init bash \
    && sudo rm -f /tmp/miniconda3.sh
