## Julia functions equivalent to fixedLogitLassoInf from selectiveInference
## R package: https://github.com/selective-inference
## NOTE: functions have only been translated to julia
##       no emphasis has been put on efficiency
## NOTE: Original Authors comments labelled with OA (Ryan Tibshirani, Rob Tibshirani,
##       Jonathan Taylor, Joshua Loftus, Stephen Reid)
## Claudia October 2017


## to do:
## - test later with the same example in R to compare to R results: still not working
## - can we make them more efficient? @profile

## ---------------------------------------------------------------------------------------
## types
## ---------------------------------------------------------------------------------------

struct fixedLogitLasso
  lambda::Number
  pv::AbstractArray
  ci::AbstractArray
  tailarea::AbstractArray
  vlo::AbstractArray
  vup::AbstractArray
  sd::AbstractArray
  vars::AbstractArray
  alpha::Number
  coef::AbstractArray
  zscore::AbstractArray
  MM::AbstractArray
end


function Base.show(io::IO, out::fixedLogitLasso)
    vars = find(x->x==true,out.vars)
    print(io,"Post selection for logistic lasso\n")
    print(io,string("Testing results at lambda = ", out.lambda," with alpha = ", out.alpha,"\n"))
    print(io, string(length(vars)," coeficients estimated by LASSO"))
    print(io,"\n\n")

    dat = DataFrame(Var=vars, Coef =out.coef, Zscore = out.zscore, Pvalue=out.pv,
                    LowConfPt=out.ci[:,1], UpConfPt=out.ci[:,2], LowTailArea = out.tailarea[:,1],
                    UpTailArea = out.tailarea[:,2]) #vlo=out.vlo, vup = out.vup)
    @show dat
end


## ---------------------------------------------------------------------------------------
## auxiliary functions
## ---------------------------------------------------------------------------------------

function checkXY(x,y)
    size(x,2) == 0 && error("There must be at least one predictor [must have ncol(x) > 0]")
    checkcols(x) && warn("x should not have duplicate columns")
    size(y,1) == 0 && error("There must be at least one data point [must have length(y) > 0]")
    size(y,1)!= size(x,1) && error("Dimensions don't match [length(y) != nrow(x)]")
end

# OA: Make sure that no two columms of A are the same
# (this works with probability one).
function checkcols(X)
    b = randn(size(X,1))
    a = sort(X' * b)
    return any(diff(a) .== 0)
end


# OA: Compute the truncation interval and SD of the corresponding Gaussian
function mylimits(Z, A, b, eta; Sigma=nothing)
    target_estimate = eta' * Z

    if (maximum(A * Z - b) > 0)
        warn("Constraint not satisfied. A * Z should be elementwise less than or equal to b")
    end
    n = length(Z)
    if (isa(Sigma,Void))
        Sigma = diagm(fill(1, n))
    end

    # OA: compute pvalues from poly lemma:  full version from Lee et al for full matrix Sigma
    var_estimate = eta' * (Sigma * eta)
    cross_cov = Sigma * eta

    resid = (eye(n) - (cross_cov ./ var_estimate * eta')) * Z
    rho = A * (cross_cov ./ var_estimate)
    vec = (b - A * resid) ./ rho

    rholess = vec[rho .< 0]
    rhogreat = vec[rho .> 0]
    vlo = length(rholess) > 0 ? maximum(rholess) : nothing
    vup = length(rhogreat) > 0 ? minimum(rhogreat) : nothing

    sd = sqrt(var_estimate)
    return (vlo, vup, sd, target_estimate)
end

## Z=bbar
## A=A1
## b=b1
## eta=vj
## Sigma=MM
## ## after mylimits:
## targetest = target_estimate

## returns (pvalue,limits of Z in [vlo,vup], sd, target estimate)
function mypvalues(Z, A, b, eta; Sigma=nothing, null_value=0)
    vlo,vup,sd,targetest = mylimits(Z, A, b, eta, Sigma=Sigma)
    return pvaluebase(vlo,vup,sd,targetest, null_value=null_value)
end

##null_value=0.0
## after pvaluebase
##pv = pp
function pvaluebase(vlo,vup,sd,targetest; null_value=0)
    pv = tnormSurv(targetest, null_value, sd, vlo, vup)
    return (pv, vlo, vup, sd, targetest)
end

## zz=targetest
## mean=null_value
## aa=vlo
## bb=vup
# Returns Prob(Z>zz | Z in [aa,bb])
function tnormSurv(zz, mean::Number, sd, aa, bb)
    zz = max(min(zz,bb),aa)

    if(mean == -Inf)
        return 0.0
    elseif(mean == Inf)
        return 1.0
    end
    pp = mpfrTnormSurv(zz,mean,sd,aa,bb)

    if(isna(pp) || isnan(pp))
        pp = brycTnormSurv(zz,mean,sd,aa,bb)
    end
    return pp
end

## mean=Vector(param_grid[2:3])
## fixit: can we combine the two tnormSurv functions?
function tnormSurv(zz, mean::Union{Vector{Int},Vector{Float64}}, sd, aa, bb)
    zz = max(min(zz,bb),aa)

    # OA: Check silly boundary cases
    p = fill(0.0,length(mean))
    p[mean.==-Inf] = 0
    p[mean.==Inf] = 1

    # OA: Try the multi precision floating point calculation first
    o = isfinite.(mean)
    mm = mean[o]

    pp = mpfrTnormSurv(zz,mm,sd,aa,bb)


    # OA: If there are any NAs, then settle for an approximation
    oo = isna.(pp) .| isnan.(pp)
    if (any(oo))
        pp[oo] = brycTnormSurv(zz,mm[oo],sd,aa,bb)
    end
    p[o] = pp
    return p
end

# Returns Prob(Z>zz | Z in [aa,bb]), works for vector mean
function mpfrTnormSurv(zz, mean, sd, aa, bb)
    # just use standard floating point calculations
    zz = (zz-mean)/sd
    aa = (aa-mean)/sd
    bb = (bb-mean)/sd

    d = Normal(0,1)
    return (pdf(d,bb)-pdf(d,zz)) ./ (pdf(d,bb)-pdf(d,aa))
end

# OA: Returns Prob(Z>zz | Z in [aa,bb]), where mean can be a vector, based on
# A UNIFORM APPROXIMATION TO THE RIGHT NORMAL TAIL INTEGRAL, W Bryc
# Applied Mathematics and Computation
# Volume 127, Issues 23, 15 April 2002, Pages 365--374
# https://math.uc.edu/~brycw/preprint/z-tail/z-tail.pdf
function brycTnormSurv(zz, mean, sd, aa, bb)
    zz = (zz-mean)/sd
    aa = (aa-mean)/sd
    bb = (bb-mean)/sd
    n = length(mean)

    term1 = n == 1 ? fill(exp.(zz.*zz),n) : exp.(zz.*zz)
    o = aa .> -Inf
    term1[o] = ff(aa[o]) .* exp.(-(aa[o] .* aa[o] - zz[o] .* zz[o]) ./2)

    term2 = fill(0.0,n)
    oo = bb .< Inf
    term2[oo] = ff(bb[oo]) .* exp.(-(bb[oo] .* bb[oo] - zz[oo] .* zz[oo]) ./2)
    p = (ff(zz)-term2)./(term1-term2)

    # OA: Sometimes the approximation can give wacky p-values,
    # outside of [0,1] ..
    #p[p<0 | p>1] = NA
    p = min.(1,max.(p,0))
    return length(p) == 1? p[1] : p ##dirty fix, need to change
end

function ff(zz)
  return (zz.*zz + 5.575192695*zz + 12.7743632) ./
         (zz.*zz.*zz*sqrt(2*pi) + 14.38718147*zz.*zz + 31.53531977*zz + 2*12.77436324)
end




## function myinterval(Z, A, b, eta; Sigma=nothing, alpha=0.1,
##                     gridrange=[-100,100],
##                     gridpts=100,
##                     griddepth=2,
##                     flip=false)

##     vlo,vup,sd,targetest = mylimits(Z, A, b, eta, Sigma)

##     return intervalbase(vlo,vup,sd,targetest,alpha=alpha,gridrange=gridrange,griddepth=griddepth,flip=flip)
## end

if(false)
    sd=sd0
    targetest=te0
    vlo=vlo0
    vup=vup0
    alpha=alpha
    gridrange=[-100,100]
    gridpts=100
    griddepth=2
    flip=false
end
## deleted intervalbase, because we do not need the two functions,
## it does not make sense to call mylimits twice (one in mypvalues and one in myinterval)
function myinterval(vlo,vup,sd,targetest; alpha=0.1,
                        gridrange=[-100,100],
                        gridpts=100,
                        griddepth=2,
                        flip=false)

    # OA: compute sel intervals from poly lemmma, full version from Lee et al for full matrix Sigma
    param_grid = linspace(gridrange[1] * sd, gridrange[2] * sd, gridpts)

    function pivot(param)
        tnormSurv(targetest, param, sd, vlo, vup)
    end

    interval = gridSearch(param_grid, pivot, alpha/2, 1-alpha/2, gridpts=gridpts, griddepth=griddepth)
    tailarea = [pivot(interval[1]), 1- pivot(interval[2])]

    if (flip)
        interval = [-interval[2],-interval[1]]
        tailarea = [tailarea[2],tailarea[1]]
    end

    return interval,tailarea
end



if(false)
    grid=param_grid
    fun=pivot
    val1=alpha/2
    val2=1-alpha/2
end
# OA: Assuming that grid is in sorted order from smallest to largest,
# and vals are monotonically increasing function values over the
# grid, returns the grid end points such that the corresponding
# vals are approximately equal to {val1, val2}
function gridSearch(grid, fun, val1, val2; gridpts=100, griddepth=2)
    n = length(grid)
    vals = fun(Vector(grid))

    ii = find(x -> x >= val1,vals)
    jj = find(x -> x <= val2,vals)
    if (length(ii)==0)
        return [grid[n],Inf] # All vals < val1
    end
    if (length(jj)==0)
        return [-Inf,grid[1]]  # All vals > val2
    end
    # OA: RJT: the above logic is correct ... but for simplicity, instead,
    # we could just return c(-Inf,Inf)

    i1 = minimum(ii)
    i2 = maximum(jj)
    if (i1==1)
        lo = -Inf
    else
        lo = gridBsearch(grid[i1-1],grid[i1],fun,val1,gridpts=gridpts,griddepth=griddepth-1,below=true)
    end
    if (i2==n)
        hi = Inf
    else
        hi = gridBsearch(grid[i2],grid[i2+1],fun,val2,gridpts=gridpts,griddepth=griddepth-1,below=false)
    end
    return [lo,hi]
end

## left=grid[i2]
## right=grid[i2+1]
## fun=pivot
## val=val2
## griddepth=griddepth-1
## below=false
# OA: Repeated bin search to find the point x in the interval [left, right]
# that satisfies f(x) approx equal to val. If below=TRUE, then we seek
# x such that the above holds and f(x) <= val; else we seek f(x) >= val.
function gridBsearch(left, right, fun, val; gridpts=100, griddepth=1, below=true)
    n = gridpts
    depth = 1

    while (depth <= griddepth)
        grid = linspace(left,right,n)
        vals = fun(Vector(grid))

        if (below)
            ii = find(x-> x >= val, vals)
            length(ii) != 0 || error("we have all vals<val, and this should not happen")
            i0 = minimum(ii)
            i0 != 1 || error("we have all vals>val, and this should not happen")
            left = grid[i0-1]
            right = grid[i0]
        else
            ii = find(x-> x <= val, vals)
            length(ii) != 0 || error("all vals>val, and this should not happen")
            i0 = maximum(ii)
            i0 != n || error("all vals<val, and this should not happen")
            left = grid[i0]
            right = grid[i0+1]
        end
        depth += 1
    end

    return below? left: right
end

## to test manually
if(false)
    alpha=0.05
    tolbeta=0.00001
    tolkkt=0.1
end

## ----------------------------------------------------------------------------------------------
## main post selection lasso function for logistic data
## ----------------------------------------------------------------------------------------------
## same alpha default as the R version
function fixedLogitLassoInference(X,y,intercept,beta,lambda;alpha=0.1::Number, tolbeta=1e-5::Number, tolkkt=0.1::Number)
    checkXY(X,y)
    n=length(y)
    p=size(X,2)
    nvar=sum(beta[2:end].!=0)
    ## initialize:
    vpv=fill(-1.0, nvar)
    vvlo=fill(-1.0, nvar)
    vvup=fill(-1.0, nvar)
    vsd=fill(-1.0, nvar)
    vci=fill(-1.0,nvar,2)
    vtailarea=fill(-1.0,nvar,2)

    # OA: do we need to worry about standardization?
    #  obj = standardize(x,y,TRUE,FALSE)
    #  x = obj$x
    #  y = obj$y

    m = beta.!=0  #active set
    vars = (abs.(beta) .> (tolbeta ./ sqrt.(sum(X .* X,1)))')[:,1]
    if(sum(vars)==0)
        error("Empty model")
    end
    m = m .& vars

    bhat = Vector(beta[m])
    unshift!(bhat,intercept) ## intercept + active terms
    DEBUG && @show bhat
    s2 = sign.(bhat)
    lam2m = diagm(vcat(0.0,fill(lambda,sum(m)))) ## create diag matrix with lambda
    DEBUG && @show lam2m

    xxm = hcat(fill(1,size(X,1)),X[:,m]) ## design matrix
    DEBUG && @show xxm
    etahat = xxm * bhat
    DEBUG && @show etahat
    prhat = exp.(etahat) ./ (1 + exp.(etahat))
    DEBUG && @show prhat
    ww = prhat.*(1-prhat)
    DEBUG && @show ww

    #check KKT
    z = etahat+(y-prhat)./ww
    DEBUG && @show z
    g = (X' * diagm(1./ww)) * (z-etahat)/lambda
    DEBUG && @show g
    if( any(abs.(g) .> 1+tolkkt) )
        warn(string("Solution beta does not satisfy the KKT conditions (to within specified tolerances): ",tolkkt))
    end


    if (any(sign.(g[m]) != sign.(beta[m])))
        warn(string("Solution beta does not satisfy the KKT conditions",
                    " (to within specified tolerances): ", tolbeta))
    end

    #constraints for active variables
    a = (xxm' * diagm(1./ww)) * xxm
    DEBUG && @show a
    MM = pinv(a)
    DEBUG && @show MM
    gm = vcat(0.0,-g[m]*lambda) # OA: gradient at LASSO solution, first entry is 0 because intercept is unpenalized
                            # at exact LASSO solution it should be s2[-1]
    DEBUG && @show gm
    dbeta = MM * gm
    DEBUG && @show dbeta

    bbar = bhat - dbeta
    DEBUG && @show bbar

    A1 = -(diagm(s2))[2:end,:]
    DEBUG && @show A1
    b1 = (s2 .* dbeta)[2:end]
    DEBUG && @show b1

    tolpoly = 0.01

    if (  maximum((A1 * bbar) - b1) > tolpoly )
        error("Polyhedral constraints not satisfied; you must recompute beta more accurately.")
    end


    for jj in 1:sum(m)
        vj=fill(0.0,sum(m)+1)
        vj[jj+1]=s2[jj+1]

        # compute p-values
        pv0,vlo0,vup0,sd0,te0 = mypvalues(bbar, A1, b1, vj, Sigma=MM)
        DEBUG && @show pv0,vlo0,vup0,sd0,te0
        vpv[jj] = pv0
        vvlo[jj] = vlo0
        vvup[jj] = vup0
        vsd[jj] = sd0

        int0,tail0 = myinterval(vlo0,vup0,sd0,te0,alpha=alpha)
        DEBUG && @show int0,tail0
        vci[jj,:] = int0
        vtailarea[jj,:] = tail0
    end

    coef0 = bbar[2:end]
    se0 = sqrt.(diag(MM)[2:end])
    zscore0 = coef0./se0
    DEBUG && @show coef0,se0,zscore0

    out = fixedLogitLasso(lambda,vpv,vci,vtailarea,vvlo,vvup,vsd,vars,alpha,coef0,zscore0,MM)
    return out
end
