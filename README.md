# ScoringEngine.jl

Demonstration of a model deployment workflow.

- Reprodicible preprocessing pipeline
- Flux MLP & EvoTrees logistics models
- Preproc/model inference through HTTP.jl
- Dockerfile build receipe

Ad-hoc Monitoring examples:

- Calling the API
- Model explanation with SHAP

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

Docker images are quite big, Julia taking ~400M.

DataFrames transforms don't recognize Functors as Function. Need Functors to inherit from Functions for function like behavior in `transform(!)`. 

Vector of transformations will losse their types once imported back from BSON, resulting in error as the type is needed by DataFrames to dispatch to appropriate method. Need to restrict to single transformation function per step (no big deal since multi-thread isn't taking advantage of such vec transforms ATM). 
