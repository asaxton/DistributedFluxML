module OptOutAllReduce

using Distributed

"""

    init(_allReduceWorkers; ReduceType=Any, ChanDepth=2)

`init()` sets up `RemoteChannel()`'s and related indices for `allRecude()` to use on subsequent calls. See `allReduce` help for more details.
# Arguments
- `allReduceWorkers::Array{Int}`: Array of worker idx that will perform a allReduce between.

# Returns
- Nothing

# Throws
- Nothing
"""
function init(_allReduceWorkers; ReduceType=Any, ChanDepth=2)
    totAllReduceChan = Dict(w_i => RemoteChannel(()->Channel{ReduceType}(ChanDepth), w_i)
                         for w_i in _allReduceWorkers)
    n_w = length(_allReduceWorkers)
    w_l = sort(_allReduceWorkers)
    allReduce_chan_fut = []
    for m_id in _allReduceWorkers
        w_i_i_map = Dict(w_i => i for (i,w_i) in zip(0:(n_w-1), w_l))
        i_w_i_map = Dict(i => w_i for (i,w_i) in zip(0:(n_w-1), w_l))
        _right_id = i_w_i_map[(w_i_i_map[ m_id] + 1) % n_w]
        fut = @spawnat m_id global right_id = _right_id
        push!(allReduce_chan_fut, fut)
        fut = @spawnat m_id global allReduceChan = Dict(m_id => totAllReduceChan[m_id],
                                                        _right_id => totAllReduceChan[_right_id])
        push!(allReduce_chan_fut, fut)
        fut = @spawnat m_id global allReduceWorkers = _allReduceWorkers
        push!(allReduce_chan_fut, fut)
        fut = @spawnat m_id global finished_init = true
        push!(allReduce_chan_fut, fut)
    end
    for fut in allReduce_chan_fut
        wait(fut)
    end
end

"""

    allReduce(func, dat)

`allRecude()` performes the All Reduce collective, with an "Opt Out" option, accross the group of workers passed to init(). If a worker calls `allReduce()` passing the symbol `:Skip` to `dat` that worker will perform a "No Op" on the reduce. Lastly, values returned are a 2 item tuple with the first item is the value of the reduce and the second is number of items that were reduced. 

# Examples

```
using Distributed
p = addProc(3)

@everywhere using DistributedFluxML

DistributedFluxML.OptOutAllReduce.init(p)

mock_vals = [1,2,3]

allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
            for (v,w) in zip(mock_vals,p)]
[fetch(fut) for fut in allR_fut] # [(6,3), (6,3), (6,3)]

mock_vals = [1,:Skip,3]

allR_fut = [@spawnat w DistributedFluxML.OptOutAllReduce.allReduce(+, v)
            for (v,w) in zip(mock_vals,p)]
[fetch(fut) for fut in allR_fut] # [(4,2), (4,2), (4,2)]
```

# Arguments
- `func(x,y)::Function`: Aggrigation function is use in reduce. `func()` must be associative to produce meaninful results.
- `dat::Any`: Singlton data point to aggrigate or `Symbol` `:Skip` to skip reduce op.

# Returns
- `allReduce(...)::Tuple{Any, Int64}`

# Throws
- `ERROR`: If `init()` was not run or finish correctly, thows an error message telling you so
"""
function allReduce(func, dat)
    if !@isdefined finished_init
        throw("""allReduce is not being called on a worker which """*
              """init() was run. """*
              """See allReduce() docString for help. myid() $(myid()) """)
    end
    if !finished_init
        throw("init() did not run successfully for $(myid())")
    end
    countD = dat == :Skip ? 0 : 1
    pDat = countD
    n_rWorkers = length(allReduceWorkers)
    for _i in 1:(n_rWorkers-1)
        put!(allReduceChan[myid()], pDat)
        tDat = take!(allReduceChan[right_id])
        pDat = tDat + countD
    end
    put!(allReduceChan[myid()], pDat)
    numReduceSteps = take!(allReduceChan[right_id])

    if numReduceSteps == 0
        return (nothing, 0)
    end
    
    if dat == :Skip
        tDat = nothing
        for _i in 1:(n_rWorkers)
            tDat = take!(allReduceChan[right_id])
            put!(allReduceChan[myid()], tDat)
        end
        return (tDat, numReduceSteps)
    else
        pDat = dat
        tDat = nothing
        for _i in 1:(numReduceSteps-1)
            put!(allReduceChan[myid()], pDat)
            tDat = take!(allReduceChan[right_id])
            pDat = func(tDat,dat)
        end
        for _i in 1:(n_rWorkers-numReduceSteps+1)
            put!(allReduceChan[myid()], pDat)
            tDat = take!(allReduceChan[right_id])
            pDat = tDat
        end
        return (pDat, numReduceSteps)
    end
end

end
