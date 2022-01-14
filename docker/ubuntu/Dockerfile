FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
ENV JULIA_VERSION 1.6.5
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y apt-utils
RUN apt-get install -y apt-transport-https ca-certificates wget

# install Julia
RUN mkdir /julia

RUN cd /julia && wget --quiet https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \ 
	tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \ 
	rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz
RUN ln -s /julia/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

# Julia packages
RUN julia -e 'using Pkg; Pkg.add("DataFrames")'
RUN julia -e 'using Pkg; Pkg.add("CSV")'
RUN julia -e 'using Pkg; Pkg.add("Arrow")'
RUN julia -e 'using Pkg; Pkg.add("BSON")'
RUN julia -e 'using Pkg; Pkg.add("JSON3")'
RUN julia -e 'using Pkg; Pkg.add("JSONTables")'
RUN julia -e 'using Pkg; Pkg.add("HTTP")'
RUN julia -e 'using Pkg; Pkg.add("Sockets")'

RUN mkdir -p /usr/src/app
COPY ../../ /usr/src/app

WORKDIR /usr/src/app

RUN julia -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()' 

ENTRYPOINT ["julia", "./bin/main.jl"]