using Test
using Pkg
Pkg.instantiate()
using Distributed
using ClusterManagers
using GLM

if Base.current_project() != nothing
    proj_path = joinpath(["/",
                          split(Base.current_project(), "/")[1:end-1]...])
    p = addprocs(SlurmManager(3),
                 time="00:30:00",
                 exeflags="--project=$(proj_path)", ntasks_per_node=1)
else
    p = addprocs(SlurmManager(3),
                 time="00:30:00",
                 ntasks_per_node=1)
end

@everywhere begin
    using DistributedFluxML
    using Serialization
    using Flux
    using Zygote
    using DataFrames
    using CSV
end

using DistributedFluxML
using Serialization
using Flux
using DataFrames
using CSV
using Zygote

status_chan = RemoteChannel(()->Channel{Any}(10000), myid())

status_array = []

stat_tsk = @async begin
    while isopen(status_chan)
        push!(status_array, take!(status_chan))
        if length(status_array) > 10000
            popfirst!(status_array)
        end
    end
end
