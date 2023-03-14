using Test
using Distributed
using ClusterManagers
using Pkg
using Flux
using Statistics
using Random
using GLM
using DataFrames
using CSV

if Base.current_project() != nothing
    proj_path = joinpath(["/",
                          splitpath(Base.current_project())[1:end-1]...])
    p = addprocs(SlurmManager(3),
                 time="00:30:00",
                 exeflags="--project=$(proj_path)", ntasks_per_node=1)
else
    p = addprocs(SlurmManager(3),
                 time="00:30:00",
                 ntasks_per_node=1)
end

ap_dir = joinpath(splitpath(Base.active_project())[1:end-1])
if "tmp" == splitpath(ap_dir)[2]
    hostNames = [@fetchfrom w gethostname() for w in p]
    ap_dir_list = [joinpath(ap_dir, d) for d in readdir(ap_dir)]
    for (w, hn) in zip(p, hostNames)

        @fetchfrom w begin
            open(`mkdir $(ap_dir)`) do f
                read(f, String)
            end
        end
    end

    for hn in hostNames
        for fpn in ap_dir_list
            open(`scp $(fpn) $(hn):$(fpn)`) do f
                read(f, String)
            end
        end
    end
end

@everywhere begin
    using Pkg
    Pkg.activate($(ap_dir))
   # Pkg.instantiate()
end

@everywhere begin
    using GLM
    using DistributedFluxML
    using Flux
    using DataFrames
    using CSV
end
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


## Load Mock data

mockData_path = joinpath(splitpath(pathof(DistributedFluxML))[1:end-2]...,"mockData")
_shard_file_list = ["iris_df_1.jlb",
                    "iris_df_2.jlb",
                    "iris_df_3.jlb"]

headers = [:sepal_l, :sepal_w, :petal_l, :petal_w, :class]
__totData = CSV.read("../mockData/iris.data", DataFrame, header=headers)
_totData = [__totData[1:50, :],
            __totData[51:100, :],
            __totData[101:end, :]
            ]

totData = Dict(i=>v for (i,v) in zip(p, _totData))

@everywhere p rawData = $(totData)[myid()]

epoch_length_worker = @fetchfrom p[1] nrow(rawData)
@everywhere p labels = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"]

@everywhere p x_array =
    Array{Float32}(rawData[:,
                           [:sepal_l, :sepal_w,
                            :petal_l, :petal_w
                            ]])

@everywhere p y_array =
    Flux.onehotbatch(rawData[:,:class],
                labels)


