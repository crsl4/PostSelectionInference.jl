## Julia translation to R package: https://github.com/selective-inference
__precompile__()

module PostSelectionInference

using DataFrames ##for isna function
global DEBUG = false #for debugging only
import Base.show

include("selective-inference.jl")

end #module
