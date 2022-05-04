using Test
using Distributed
using ClusterManagers

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
    #using Flux
    #using DataFrames
    #busing CSV
end

@testset "Opt Out All Reduce" begin
    DistributedFluxML.OptOutAllReduce.init(p)

    @test all([@fetchfrom w DistributedFluxML.OptOutAllReduce.finished_init for w in p])
    right_ids = Set([@fetchfrom w DistributedFluxML.OptOutAllReduce.right_id for w in p])
    @test right_ids == Set(p)

    test_vals = [1,2,3]
    allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
                for (v,w) in zip(test_vals,p)]
    if !all([isready(fut) for fut in allR_fut])
        sleep(2)
    end

    t = @test all([isready(fut) for fut in allR_fut])
    if isa(t, Test.Pass)
        @test [fetch(fut) for fut in allR_fut] == [(6,3), (6,3), (6,3)]
    end

    test_vals = [1,:Skip,3]
    allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
                for (v,w) in zip(test_vals,p)]
    if !all([isready(fut) for fut in allR_fut])
        sleep(2)
    end

    t = @test all([isready(fut) for fut in allR_fut])
    if isa(t, Test.Pass)
        @test [fetch(fut) for fut in allR_fut] == [(4,2), (4,2), (4,2)]
    end

    test_vals = [1,:Skip,:Skip]
    allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
                for (v,w) in zip(test_vals,p)]
    if !all([isready(fut) for fut in allR_fut])
        sleep(2)
    end

    t = @test all([isready(fut) for fut in allR_fut])
    if isa(t, Test.Pass)
        @test [fetch(fut) for fut in allR_fut] == [(1,1), (1,1), (1,1)]
    end

    test_vals = [:Skip, :Skip,3]
    allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
                for (v,w) in zip(test_vals,p)]
    if !all([isready(fut) for fut in allR_fut])
        sleep(2)
    end

    t = @test all([isready(fut) for fut in allR_fut])
    if isa(t, Test.Pass)
        @test [fetch(fut) for fut in allR_fut] == [(3,1), (3,1), (3,1)]
    end

    test_vals = [:Skip,2,:Skip]
    allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
                for (v,w) in zip(test_vals,p)]
    if !all([isready(fut) for fut in allR_fut])
        sleep(2)
    end

    t = @test all([isready(fut) for fut in allR_fut])
    if isa(t, Test.Pass)
        @test [fetch(fut) for fut in allR_fut] == [(2,1), (2,1), (2,1)]
    end

    test_vals = [:Skip,:Skip,:Skip]
    allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
                for (v,w) in zip(test_vals,p)]
    if !all([isready(fut) for fut in allR_fut])
        sleep(2)
    end

    t = @test all([isready(fut) for fut in allR_fut])
    if isa(t, Test.Pass)
        @test [fetch(fut) for fut in allR_fut] == [(nothing,0),(nothing,0),(nothing,0)]
    end

end

if false
    _shard_file_list = ["../mockData/iris_df_1.jlb",
                        "../mockData/iris_df_2.jlb",
                        "../mockData/iris_df_3.jlb"]

    shard_file_list = [joinpath(dirname(pathof(DistributedFluxML)), sf) for sf in _shard_file_list];

    data_array = [deserialize(f) for f in shard_file_list];

    labels = ["Iris-versicolor", "Iris-virginica"]

    y_array = [Flux.onehotbatch(d[:,"class"], ["Iris-versicolor", "Iris-virginica"]) for d in data_array]
    x_array = [Array(d[:, [:sepal_l,  :sepal_w,  :petal_l,  :petal_w]]) for d in data_array]
end

rmprocs(p)
