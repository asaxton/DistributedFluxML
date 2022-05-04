module DistributedFluxML

using Distributed
using Flux.Optimise: AbstractOptimiser
using Flux
using Zygote

gradType = Union{Nothing, Int, Array{Array{Float32,N} where N,1}} #typeof([l for l in θ])
ReduceType = Union{Nothing, Int, Array{Array{Float32,N} where N,1}} #typeof([l for l in θ])


function cond_put!(chan, dat)
    if chan != nothing
        return put!(chan, dat)
    else
        return nothing
    end
end



"""

    train!(loss, pars::Params, data, opt::AbstractOptimiser, workers; [cb])

  Uses a `loss` function and training `data` to improve the model's parameters according to a particular optimisation rule `opt`.

#Arguments
`loss::Function`: takes an item in data, `d`, and evauates `loss(d)`
`pars::Params`: 
`data::Dict{Int,Remote`

"""
function train!(loss, pars::Params, data,
                opt::AbstractOptimiser, trainWorkers;
                cb=()->nothing,
                status_chan=nothing,
                model_snapshot_dir="./",
                RARChanDepth=2,
                device=gpu)


    paramPassChan = RemoteChannel(()->Channel{Any}(RARChanDepth), myid())
    paramSyncTxChan = RemoteChannel(()->Channel{Any}(RARChanDepth), myid())
    paramSyncRxChan = RemoteChannel(()->Channel{Any}(RARChanDepth), myid())

    parm_fut = [@spawnat w global ModelParam = pars |> device for w in trainWorkers];


    
    
end

function train_seq!(loss, pars::Params, data,
                opt::AbstractOptimiser, workers;
                cb=()->nothing,
                status_chan=nothing,
                model_snapshot_dir="./",
                RARChanDepth=2,
                device=gpu)

    allReduceChan = Dict(w_i => RemoteChannel(()->Channel{gradType}(RARChanDepth), w_i)
                         for w_i in workers)
    paramPassChan = RemoteChannel(()->Channel{Any}(RARChanDepth), myid())
    paramSyncTxChan = RemoteChannel(()->Channel{Any}(RARChanDepth), myid())
    paramSyncRxChan = RemoteChannel(()->Channel{Any}(RARChanDepth), myid())

    parm_fut = [@spawnat w global ModelParam = pars |> device for w in workers];
    
end

function static_graph_train_loop()
end

include("OptOutAllReduce.jl")

end # module


