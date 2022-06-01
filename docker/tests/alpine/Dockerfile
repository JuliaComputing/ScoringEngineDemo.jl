FROM alpine:3.15

ENV JULIA_VERSION 1.6.5
RUN apk add --no-cache ca-certificates wget

# install Julia
RUN mkdir /julia

RUN cd /julia && wget --quiet https://julialang-s3.julialang.org/bin/musl/x64/1.6/julia-${JULIA_VERSION}-musl-x86_64.tar.gz && \ 
	tar xzf julia-${JULIA_VERSION}-musl-x86_64.tar.gz && \ 
	rm julia-${JULIA_VERSION}-musl-x86_64.tar.gz
RUN ln -s /julia/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

# Julia packages
RUN julia -e 'using Pkg; Pkg.add("HTTP")'
RUN julia -e 'using Pkg; Pkg.add("DataFrames")'
RUN julia -e 'using Pkg; Pkg.add("CSV")'
RUN julia -e 'using Pkg; Pkg.add("Arrow")'