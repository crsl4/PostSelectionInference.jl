# Post Selection Inference in Julia

[![Build Status](https://travis-ci.org/crsl4/PostSelectionInference.jl.svg)](https://travis-ci.org/crsl4/PostSelectionInference.jl)

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

```
WARNING: Solution beta does not satisfy the KKT conditions (to within specified tolerances): 0.1
Post selection for logistic lasso
Testing results at lambda = 0.0577099 with alpha = 0.1
18 coeficients estimated by LASSO

dat = 18×8 DataFrames.DataFrame
│ Row │ Var  │ Coef      │ Zscore   │ Pvalue      │ LowConfPt │ UpConfPt │ LowTailArea │ UpTailArea │
├─────┼──────┼───────────┼──────────┼─────────────┼───────────┼──────────┼─────────────┼────────────┤
│ 1   │ 6    │ 0.231436  │ 11.2027  │ 1.5555e-10  │ 0.180643  │ 0.282241 │ 0.0498557   │ 0.0488584  │
│ 2   │ 836  │ -0.313391 │ -14.519  │ 2.181e-22   │ 0.253927  │ 0.405006 │ 0.0476926   │ 0.0441273  │
│ 3   │ 1274 │ -0.251692 │ -12.3996 │ 0.0282467   │ 0.039557  │ 0.245419 │ 0.0498095   │ 0.0468426  │
│ 4   │ 3631 │ 0.226974  │ 10.6836  │ 0.0012874   │ 0.120738  │ 0.218716 │ 0.0495266   │ 0.0434461  │
│ 5   │ 4540 │ 0.265718  │ 13.2502  │ 0.345798    │ -0.482676 │ 0.252285 │ 0.0499511   │ 0.0493574  │
│ 6   │ 4546 │ -0.271242 │ -13.2715 │ 0.0157925   │ 0.0744446 │ 0.26504  │ 0.0499879   │ 0.0449219  │
│ 7   │ 4803 │ 0.226952  │ 11.4849  │ 2.88696e-6  │ 0.164725  │ 0.315537 │ 0.0497685   │ 0.0462282  │
│ 8   │ 5772 │ -0.318026 │ -15.3722 │ 3.93541e-15 │ 0.267022  │ 0.387763 │ 0.0482941   │ 0.0489693  │
│ 9   │ 6185 │ 0.199402  │ 9.14568  │ 2.94442e-6  │ 0.139479  │ 0.256935 │ 0.0487711   │ 0.0470105  │
│ 10  │ 6277 │ 0.166598  │ 7.8605   │ 0.0130619   │ 0.0495205 │ 0.158941 │ 0.0497953   │ 0.0419885  │
│ 11  │ 6903 │ -0.254149 │ -12.6187 │ 4.01423e-8  │ 0.197482  │ 0.239403 │ 0.0496868   │ 0.0237093  │
│ 12  │ 7088 │ -0.224877 │ -10.7906 │ 1.04922e-10 │ 0.173721  │ 0.286842 │ 0.0483763   │ 0.0498232  │
│ 13  │ 7194 │ 0.189274  │ 9.41536  │ 0.00845453  │ 0.0683012 │ 0.182342 │ 0.049707    │ 0.0429388  │
│ 14  │ 7336 │ 0.256622  │ 11.6985  │ 8.96134e-16 │ 0.199421  │ 0.313568 │ 0.0488022   │ 0.0487606  │
│ 15  │ 7659 │ 0.220502  │ 10.3783  │ 2.1785e-6   │ 0.158465  │ 0.205722 │ 0.0488162   │ 0.0318351  │
│ 16  │ 7668 │ 0.318928  │ 15.7256  │ 1.94364e-22 │ 0.265487  │ 0.387988 │ 0.0489357   │ 0.0463162  │
│ 17  │ 8803 │ 0.268214  │ 12.8333  │ 0.00821775  │ 0.0991579 │ 0.261649 │ 0.0498012   │ 0.0460855  │
│ 18  │ 9368 │ -0.218595 │ -10.5464 │ 0.00011478  │ 0.139787  │ 0.276402 │ 0.0491593   │ 0.0467489  │
```