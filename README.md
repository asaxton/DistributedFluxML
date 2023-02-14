# jlDistributedFluxML

This package is to be used with FluxML to train, evaluate (inference), and analyze models on a distributed cluster. At the moment only the Slurm cluster manager has been tested.

## Getting started
Comming soon

### Training
These examples assumes that you have already partitioned the data into multiple `DataFrame`s and serialized them using `Serialization` package  into `shard_file_list`
```
    using Distributed
    p = addProc(3)

    @everywhere using DistributedFluxML


    batch_size=8
    epochs = 50

    deser_fut = [@spawnat w global rawData = deserialize(f)
                 for (w, f) in zip(p, shard_file_list)]
    for fut in deser_fut
        wait(fut)
    end

    @everywhere p labels = ["Iris-versicolor", "Iris-virginica", "Iris-setosa"]

    @everywhere p x_array =
        Array(rawData[:,
                      [:sepal_l, :sepal_w,
                       :petal_l, :petal_w
                       ]])

    @everywhere p y_array =
        Flux.onehotbatch(rawData[:,"class"],
                         labels)

    @everywhere p dataChan = Channel(1) do ch
        n_chunk = ceil(Int,size(x_array)[1]/$batch_size)
        x_dat = Flux.chunk(transpose(x_array), n_chunk)
        y_dat = Flux.chunk(y_array, n_chunk)
        for epoch in 1:$epochs
            for d in zip(x_dat, y_dat)
                put!(ch, d)
            end
        end
	put!(ch, :End)
    end

    @everywhere p datRemChan = RemoteChannel(() -> dataChan, myid())

    trainWorkers_shift = circshift(p, 1)
    # ^^^ shift workers to reuse workers as ^^^
    # ^^^ remote data hosts ^^^
    datRemChansDict = Dict(k => @fetchfrom w datRemChan for (k,w) in zip(p, trainWorkers_shift))

    loss_f = Flux.Losses.logitcrossentropy
    opt = Flux.Optimise.ADAM(0.001)

    model = Chain(Dense(4,8),Dense(8,16), Dense(16,3))

    empty!(status_array)
    DistributedFluxML.train!(loss_f, model, datRemChansDict, opt, p; status_chan)
```

### Custom Gradiant Calc

Argument `grad_calc` in `train!` is meant for novel nomalization schemes. For example, if your `DataChannel` returns a 3 touple, say `(x, s, y)`, a desired grad calc coule be

```
function node_norm_grad_calc(xsy, loss_f, model; device=gpu)
    x,s,y = xsy
    loss_f(model(x), y, agg=sum)/s
end
```


### Opt Out All Reduce
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
