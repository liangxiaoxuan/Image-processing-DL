ARG GPU
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y locales rxvt-unicode-256color xauth
RUN apt-get update --fix-missing && apt-get install -y make autoconf automake pkg-config gcc \
    build-essential g++ python3.5 python3.5-dev python3.5 git wget libncurses5-dev \
    lua5.1 liblua5.1-dev libperl-dev cmake libgnome2-dev libgnomeui-dev \
    libgtk2.0-dev libatk1.0-dev libbonoboui2-dev fonts-powerline\
    libcairo2-dev libx11-dev libxpm-dev libxt-dev python-dev python3-dev \
    ruby-dev mercurial python3-pip python3-venv exuberant-ctags libevent-dev \
    tzdata fortune cowsay inetutils-ping libclang1 htop
RUN python3 -m pip install --upgrade pip
#Install IPython
RUN python3 -m pip install powerline-status powerline-gitstatus ipython matplotlib ipykernel jupyter jupyter_console notedown pillow
EXPOSE 8888
# alias python -> python3
RUN mkdir -p /opt/alias && echo '#!/bin/bash' > /opt/alias/python && echo 'python3 "$@"' >> /opt/alias/python && chmod +x /opt/alias/python
RUN echo '#!/bin/bash' > /opt/alias/ipython && echo 'ipython3 "$@"' >> /opt/alias/ipython && chmod +x /opt/alias/ipython
ENV PATH /opt/alias:$PATH
# Dependencies
RUN apt-get update && apt-get install -y \
  g++ \
  python3.5 \
  python3-pip python3-dev git

RUN python3 -m pip install --upgrade matplotlib jupyter-tensorboard opencv-python
WORKDIR /usr/local/lib/python3.5/dist-packages
RUN ctags -R .

# Install python libraries
#RUN if [[ -n "$GPU"  ]] ; then python3 -m pip install --upgrade tensorflow-gpu==1.12.0 keras==2.2.5; else python3 -m pip install --upgrade tensorflow==1.12.0 ; fi
RUN if [[ -n "$GPU" ]] ; then python3 -m pip install --upgrade tensorflow-gpu==1.13.1 keras==2.3.0; else python3 -m pip install --upgrade tensorflow==1.13.1 keras==2.3.0 ; fi
RUN pip install beautifulsoup4 colorama cython googledrivedownloader hickle \
     image-classifiers==1.0.0b1 imageio imagesize lxml  nu
