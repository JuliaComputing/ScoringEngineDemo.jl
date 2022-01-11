# ScoringEngine.jl

Demonstration of a model deployment workflow.

- Reprodicible preprocessing pipeline
- Flux MLP & EvoTrees logistics models
- Preproc/model inference through HTTP.jl
- Dockerfile build receipe

Ad-hoc Monitoring examples:

- Calling the API
- Animation of model spread through time

Work derived from [Insurance-pricing-game](https://www.aicrowd.com/challenges/insurance-pricing-game) challenge.

Building image:
```
docker build . -f docker\\ubuntu\\Dockerfile -t scoring:test
```

Starting the scoring engine:
```
docker run -it -d -p 8008:8008 -t scoring:test 
```

### Notes

Docker images are quite bug. Julia taking ~400M. Not super prod appealing. 

DataFrames transforms won't recignize Functors as Function. Need Functors to inherit from Functions to work. 

Vector of transformations will losse their types once imported back from BSON, resulting in error as the type is needed by DataFrames to dispatch to appropriate method. 
