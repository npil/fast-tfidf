FROM mcr.microsoft.com/devcontainers/base:debian
# FROM mcr.microsoft.com/devcontainers/base:jammy

ARG USER=vscode

RUN DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && apt install -y \
        # dev tools
        git \
        make\
        vim \
        # python build deps
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        liblzma-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        tk-dev \
        uuid-dev \
        zlib1g-dev

# Python and poetry installation
USER $USER
ARG HOME="/home/$USER"
ARG PYTHON_VERSION=3.13

ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${HOME}/.local/bin:$PATH"

RUN echo "done 0" \
    && curl https://pyenv.run | bash \
    && pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false \
    && echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc
