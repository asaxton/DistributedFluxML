include("globalTestInit.jl")
include("OptOutAllReduce.jl")
include("trainModel.jl")
rmprocs(p)
