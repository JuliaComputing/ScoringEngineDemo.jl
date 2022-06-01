FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
ENV JULIA_VERSION 1.6.5
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y apt-utils
RUN apt-get install -y apt-transport-https ca-certificates wget
RUN apt-get install -y dirmngr gnupg software-properties-common build-essential

# install R
RUN apt-key adv --keyserver hkps://keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
RUN apt-get update && apt-get -y install r-base