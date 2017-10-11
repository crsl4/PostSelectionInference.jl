# PostSelectionInference.jl
Julia translation to [selective-inference/R-software](https://github.com/selective-inference/R-software)

```julia
## Simulating data
srand(1234)
X = randn(500,10000)
y = convert(Array{Float64,1},rand(500).>0.3)

## Fitting logistic Lasso
using Lasso
f2=@time fit(LassoPath,X,y,Bernoulli(),LogitLink())

## Choosing one model for post-selection (ideally, we do cross validation)
beta = f2.coefs[:,7]
lambda = 0.0577099
intercept = f2.b0[7]

## We get post-selection pvalues and CI
out = fixedLogitLassoInference(X,y,intercept,beta,lambda);
@show out
```