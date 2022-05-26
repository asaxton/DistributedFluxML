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
