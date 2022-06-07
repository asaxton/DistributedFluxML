using DistributedFluxML
include("globalTestInit.jl")
include("OptOutAllReduce.jl")
include("trainModel.jl")
include("evalModel.jl")
rmprocs(p)
