# ScoringEngine.jl

Demonstration of a model deployment workflow.

- Reproducible preprocessing pipeline
- Flux MLP & EvoTrees logistics models
- Preproc/model inference through HTTP.jl
- Stipple interactive dashboard for data exploration and model explainability

Work derived from [Insurance-pricing-game](https://www.aicrowd.com/challenges/insurance-pricing-game) challenge.

### Running on JuliaHub

Add `ScoringEngineDemo.jl` as a Custom App. 
Launch the app, specifying port 8000. This will run the Stipple Dashboard.

### Docker

Building and running API service container:

```
docker build . -f docker/api/Dockerfile -t scoring:api
docker run -it -d --rm -p 8008:8008 -t scoring:api
```

Building and running Stipple dashboard container:

```
docker build . -f docker/stipple/Dockerfile -t scoring:stipple
docker run -it -d --rm -p 8000:8000 -t scoring:stipple
```

### Notes

Docker images are quite big, Julia taking ~400M.

DataFrames transforms don't recognize Functors as Function. Need Functors to inherit from Functions for function like behavior in `transform(!)`. 

Vector of transformations will losse their types once imported back from BSON, resulting in error as the type is needed by DataFrames to dispatch to appropriate method. Need to restrict to single transformation function per step (no big deal since multi-thread isn't taking advantage of such vec transforms ATM). 
