## test functions for the fixedLogitLassoInference function
## and compare to R results
## Claudia Octuber 2017

include("../src/selective-inference.jl")
## using SnpArrays
## datafolder = "/Users/Clauberry/Documents/gwas/data/22q/22q_files_NEW/"
## chr22 = SnpArray(string(datafolder,"bedfiles/22q-chr22"))
## maf, minor_allele, missings_by_snp, missings_by_person = summarize(chr22)
## chr22 = chr22[:, maf.>=0.05]
## r = 12345
## srand(r)
## ind = rand(1:size(chr22,2),10000)
using DataFrames
## chr22sub = chr22[:,ind]
dat = readtable("test.txt", header=true);
y = convert(Array{Float64,1},dat[:,1]) ##1st column: response
X = convert(Array{Float64,2},dat[:,2:end]) ##10,000 SNPs
## R selective-inference does not allow X to have repeated columns, so
## for this example, let's remove them

## use drop duplicates, need to find this function
## DataFrames removes duplicated rows, so:
df = DataFrame(X');
unique!(df);
size(df) == (8867,519) || error("wrong size of matrix with unique columns")
X2 = convert(Array{Float64,2},df) ##SNPs
X2 = X2'
size(X2) == (519,8867) || error("wrong size of matrix with unique columns")

using Lasso
f2=@time fit(LassoPath,X2,y,Bernoulli(),LogitLink())
beta = f2.coefs[:,9]
lambda = 0.0615759
intercept = f2.b0[9]
out = fixedLogitLassoInference(X2,y,intercept,beta,lambda)
@show out

## now we can use RCall to send the matrix and everything to R and test over there.
## send things around with R with @rput @rget
## we need that beta_hat includes the intercept:
beta_hat = Vector(deepcopy(beta))
unshift!(beta_hat,intercept)
using RCall
@rput X2
@rput y
@rput beta_hat
@rput lambda


## I did not use selective-inference-currentCRAN for the code, so
## I git cloned the repo and used the other R functions to test
R"""
library(selectiveInference)
#source("/Users/Clauberry/software/selective-inference/R-software/selectiveInference/R/funs.common.R")
#source("/Users/Clauberry/software/selective-inference/R-software/selectiveInference/R/funs.fixedLogit.R")
#source("/Users/Clauberry/software/selective-inference/R-software/selectiveInference/R/funs.fixed.R")
#source("/Users/Clauberry/software/selective-inference/R-software/selectiveInference/R/funs.inf.R")
outR = fixedLassoInf(X2,y,beta_hat,lambda,family="binomial")
#out=fixedLogitLassoInf(X,y,beta,lambda,alpha=alpha, type="partial", tol.beta=tol.beta, tol.kkt=tol.kkt,
#                           gridrange=gridrange, bits=bits, verbose=verbose,this.call=this.call)
"""
@rget outR

using Base.Test
at = 1e-3
@test out.MM ≈ outR[Symbol("info.matrix")] atol=at
@test out.tailarea ≈ outR[:tailarea] atol=at
@test out.coef ≈ outR[:coef0] atol=at
@test out.zscore ≈ outR[:zscore0] atol=at
@test out.pv ≈ outR[:pv] atol=at
@test out.vup ≈ outR[:vup] atol=at
@test out.vlo ≈ outR[:vlo] atol=at
@test out.ci ≈ outR[:ci] atol=at
@test out.sd ≈ outR[:sd] atol=at
