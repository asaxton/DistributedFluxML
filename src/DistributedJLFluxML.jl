module DistributedJLFluxML

using Distributed
using Flux.Optimise: AbstractOptimiser
using Flux
# using Zygote: @nograd # has been useful in the past, keeping it here incase.

function cond_put!(chan, dat)
    if chan != nothing
        return put!(chan, dat)
    else
        return nothing
    end
end

function makeStatDict(status_name::String; kwargs...)
    Dict(:statusName=>status_name,
         :myid=>myid(),
         :hostname=>gethostname(),
         :timestamp=>time(),
         kwargs...)
end

include("OptOutAllReduce.jl")


"""
Leaving this hear for future work. Looks like theres a problem with Distributed & Threads that
might be breaking this.

See https://github.com/JuliaLang/julia/issues/32677  and
https://github.com/JuliaLang/julia/pull/33555

When Julia 1.8 is realeased, readdress this
"""
function build_data_loader_from_RemoteChannel(rChan)
    chan = Channel() do ch
        while true
            try
                res = take!(rChan)
                put!(ch, res)
            catch err
                if isa(err, InvalidStateException) & (err.state == :closed)
                    break
                end
                if isa(err, RemoteException) &
                    isa(err.captured.ex, InvalidStateException) &
                    (err.captured.ex.state == :closed)
                    break
                end
                #rethrow()
            end
        end
    end
end

"""

    std_grad_calc(xy, loss_f, model; device)

  Standard combination on loss_f, model, input and output to be used in

```
  l = Flux.gradient(θ) do
    ...
  end
```

# Arguments
- `xy::Tuple`: A tuple with the first item as the model input and second item the expected model output
- `loss_f::Function`: evauates `loss_f(y̅, y)`
- `model::Chain`: The model to be trained


# Returns
- `Float`: the value of the loss calculated by loss_f
"""
function std_grad_calc(xy, loss_f, model; device=gpu)
    x, y = xy
    xs = size(x)
    loss_fac_cpu = Float32(reduce(*, xs))
    loss_fac_gpu = loss_fac_cpu |> device
    loss_f(model(x), y, agg=sum)/loss_fac_gpu
end

"""
    do_train_on_remote()

Not documented. Called though @swpanat in `train!()`. No one whould be calling this directly.
"""
function do_train_on_remote(loss_f, model, data, opt; status_chan=nothing,
                            saved_model_dir=nothing,
                            master=myid(),
                            cb=()->nothing,
                            device=gpu,
                            size_data_load_buff=2,
                            save_on_step_cb=st -> true,
                            grad_calc=std_grad_calc)

    if (master == myid()) & (saved_model_dir != nothing)
        global save_model_f = open(joinpath(saved_model_dir, "savedModelParam.jlb"), "w")
    end

    θ = Flux.params(model)
    gr_share = [zeros(Float32, size(l)) for l in θ]
    acum_gr_share = copy(gr_share)

    # TODO: In future versions of Julia, we may not need to handle
    # remote channels ourselves 5/13/22:saxton
    # https://github.com/JuliaLang/julia/pull/33555
    # https://github.com/JuliaLang/julia/pull/41966
    #if isa(data, RemoteChannel)
    #    global data_dl = build_data_loader_from_RemoteChannel(data)
    #else
    #    global data_dl = data
    #end
    # See docstring in build_data_loader_from_RemoteChannel
    data_dl = Channel(size_data_load_buff, spawn=true) do ch
        while true
            t_dat = take!(data)
            if typeof(t_dat) == Symbol
                if t_dat == :End
                    put!(ch, t_dat)
                    break
                end
            end
            gpu_data  = (_d |> device for _d in t_dat)

            put!(ch, gpu_data)
        end
    end

    loss_rep = Float32(0.0)
    step = 0

    while true
        xy = take!(data_dl)
        if typeof(xy) == Symbol
            if xy == :End
                cond_put!(status_chan, makeStatDict("do_train_on_remote.finished";
                                                    :step=>step))
                break
            end
        end
        step += 1
        cond_put!(status_chan, makeStatDict("do_train_on_remote.step";
                                            :step=>step))

        #loss_fac_cpu = Float32(230400)

        gr = Flux.gradient(θ) do
            #l = loss(xy...)
            #l = loss_f(model(x), y; agg=sum)/loss_fac_gpu
            #l = loss(x, y)/loss_fac_gpu
            l = grad_calc(xy, loss_f, model; device=device)
            loss_rep = l |> f32
            return l
        end

        cond_put!(status_chan, makeStatDict("do_train_on_remote.step.grad";
                                            :step=>step,
                                            :loss=>loss_rep,
                                            :ImNew=>3))

        for (sh, acumm_sh, p) in  zip(gr_share, acum_gr_share, gr.params)
            if gr.grads[p] != nothing
                copyto!(sh, gr.grads[p])
                _ret_sh = OptOutAllReduce.allReduce(+, sh)
                copyto!(gr.grads[p], _ret_sh[1]/_ret_sh[2])
            end
        end
        cond_put!(status_chan, makeStatDict("do_train_on_remote.step.shared";
                                            :step=>step))

        Flux.Optimise.update!(opt, θ, gr)

        cb()

        if (master == myid()) & (saved_model_dir != nothing) & save_on_step_cb(step)
            serialize(save_model_f, (step, θ))
            flush(save_model_f)
            cond_put!(status_chan, makeStatDict("do_train_on_remote.step.saved_model";
                                                :step=>step))
        end
    end

    cond_put!(status_chan, makeStatDict("do_train_on_remote.finished.wait";
                                        :step=>step))
    _ret_sh = OptOutAllReduce.allReduce(+, :Skip)

    while true
        if _ret_sh[2] ==0
            break
        end
        _ret_sh = OptOutAllReduce.allReduce(+, :Skip)
    end
    cond_put!(status_chan, makeStatDict("do_train_on_remote.finished.done";
                                        :step=>step))
    return [p |> cpu for p in θ]
end


unit_test_example_path = joinpath(splitpath(pathof(DistributedJLFluxML))[1:end-2]...,"test")
"""

    train!(loss, model, data, opt, workers;
           cb, save_on_step_cb, status_chan, save_model_dir, device,
           no_block, grad_calc=std_grad_calc)

  Uses a `loss` function and training `data` to improve the `model` parameters according to a particular optimisation rule `opt`. Runs the training loop in parellel on `workers`, agrigates the gradients through an **allReduce**, then updates the model parameters.

# Example
See test dir $(unit_test_example_path), in particular $(joinpath(unit_test_example_path, "trainModel.jl")), for more details.


Partition and load data on workers. The following example loads an already partitioned version of the iris dataset

```
    batch_size=8
    epochs = 50

    deser_fut = [@spawnat w global rawData = deserialize(f)
                 for (w, f) in zip(workersToHostData, shard_file_list)]
    for fut in deser_fut
        wait(fut)
    end
    epoch_length_worker = @fetchfrom p[1] nrow(rawData)
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
        n_chunk = ceil(Int,size(x_array)[1]/\$batch_size)
        x_dat = Flux.chunk(transpose(x_array), n_chunk)
        y_dat = Flux.chunk(y_array, n_chunk)
        for epoch in 1:\$epochs
            for d in zip(x_dat, y_dat)
                put!(ch, d)
            end
        end
    end

    @everywhere p datRemChan = RemoteChannel(() -> dataChan, myid())

    datRemChansDict = Dict(k => @fetchfrom w datRemChan for (k, w) in zip(workersToRunTrain,workersToHostData))
```

Once the data is set up for your needs, then you need to define the model, loss, optimizer and pass it to `train!`

```
    loss_f = Flux.Losses.logitcrossentropy
    opt = Flux.Optimise.ADAM(0.001)

    model = Chain(Dense(4,8),Dense(8,16), Dense(16,3))

    DistributedJLFluxML.train!(loss_f, model, datRemChansDict, opt, p)
```

# Example
Argument `grad_calc` is meant for novel nomalization schemes. For example, if your `DataChannel` returns a 3 touple, say `(x, s, y)`, a desired grad calc coule be

```
function node_norm_grad_calc(xsy, loss_f, model; device=gpu)
    x,s,y = xsy
    loss_f(model(x), y, agg=sum)/s
end
```

When `train!` returns, it will have updated the parameters of `model`.

# Arguments
- `loss::Function`: evauates `loss(y, y̅)`
- `model::Chain`: The model to be trained
- `data::Dict{Int,RemoteChannel}`: The dict key is the worker id that the remote channel will be sent to
- `opt::AbstractOptimizer`: The optimized used during training
- `workers::AbstractArray`: List of workers ids to perform training on.
- `cb::Function`: a callback that is called after optimize.update! on each training worker
- `save_on_step_cb::Function`: The training step is passed to this cb on each training iteration. If the cb returns true, a copy of the model will be saved to `saved_model_dir`
- `saved_model_dir::String`: path to directory where saved model will be placed
- `device::Function`: the device a model will be copied to on remote worker. usually `gpu` or `cpu`
- `grad_calc::Function`: a function that will be called in Flux.gradient to combine a single data sample, the model, and the loss function. See `std_grad_calc` as an example.

# Returns
- `nothing`

# Throws
- `nothing`
"""
function train!(_loss_f, _model::Chain, _data,
                _opt::AbstractOptimiser, trainWorkers;
                cb=()->nothing,
                save_on_step_cb=st -> true,
                status_chan=nothing,
                saved_model_dir=nothing,
                device=gpu,
                grad_calc=std_grad_calc)

    OptOutAllReduce.init(trainWorkers)
    
    model_fut = [@spawnat w global model = _model |> device for w in trainWorkers];
    loss_f_fut = [@spawnat w global loss_f = _loss_f for w in trainWorkers];
    opt_fut = [@spawnat w global opt = _opt for w in trainWorkers];

    wait.([model_fut..., loss_f_fut..., opt_fut...])

    train_fut = []
    for w in trainWorkers
        fut = @spawnat w do_train_on_remote(loss_f, model, _data[w], opt; status_chan=status_chan,
                                            saved_model_dir=saved_model_dir,
                                            master=trainWorkers[1],
                                            device=device,
                                            save_on_step_cb=st -> true,
                                            grad_calc=grad_calc)
        push!(train_fut, fut)
    end

    wait.(train_fut)

    θ = Flux.params(_model)
    θ_rem = fetch(train_fut[1])
    for (p1,p2) in zip(θ, θ_rem)
        copy!(p1, p2)
    end
    return train_fut
end


function do_eval_on_remote(model, data_dl;
                           status_chan=nothing, get_step=nothing,
                           device=gpu)
    cond_put!(status_chan, makeStatDict("do_eval_on_remote.start";
                                        :message=>"Starting",
                                        ))
    y=[]
    while true
        x = take!(data_dl) 
        if typeof(x) == Symbol
            if x == :End
                break
            end
        end
        x_device = x |> device
        y_cpu = model(x_device) |> cpu
        push!(y,y_cpu)
    end
    return y
end

"""
    eval_model(saved_model_dir::String, model, _data, workers;
               status_chan=nothing, get_step=nothing,
               device=gpu)

  Not tested yet. Still need to build model saving in `train!`
"""
function eval_model(saved_model_dir::String, model, _data, workers;
                    status_chan=nothing, get_step=nothing,
                    device=gpu)
    save_model_f = open(joinpath(saved_model_dir, "savedModelParam.jlb"), "r")
    open(save_mdel_f) do f
        while true
            try
                global (step, _θ) = deserialize(f)
            catch e
                if isa(e, EOFError)
                    break
                end
            end
            if step == get_step
                break
            end
        end
    end
    θ = Flux.params(model)
    for (ld, ls) in zip(θ, _θ)
        copy!(ld, ls)
    end
    res = eval_model(model, _data, workers; status_chan=status_chan, get_step=get_step,device=device)
    return res
end

"""
    eval_model(model, data, evalWorkers;
               status_chan=nothing, get_step=nothing,
               device=gpu)

  This function evaluates the model on a set of data partitioned across many workers. `eval_model` will deploy `model` and the approprate `RemoteChannel` from `data` to `evalWorkers`. There, it will call `model(x)` on the data iterated by `data[myid()]`. Finally, the results will be fetched and aggregated into a single array.

# Arguments
- `model::Chain`: The model that will be evaluated
- `data::Dict{Int,RemoteChannel}`: The dict key is the worker id that the remote channel will be sent to
- `evalWorkers::AbstractArray`: List of workers ids to perform evaluation on.
- `statusChan::RemoteChannel`: status messages and data will be placed on this channel to monitor progress
- `device::Function`: the device a model will be copied to on remote worker. usually `gpu` or `cpu`

# Returns
- `Array`: concatenated array of results from each of the workers

# Throws
- `nothing`
"""
function eval_model(_model, _data, evalWorkers; status_chan=nothing, device=gpu)

    model_fut = [@spawnat w global model = _model |> device for w in evalWorkers];
    wait.(model_fut)

    eval_fut = []
    for w in evalWorkers
        fut = @spawnat w DistributedJLFluxML.do_eval_on_remote(model, _data[w]; status_chan=status_chan)
        push!(eval_fut, fut)
    end
    wait.(eval_fut)
    res = vcat(fetch.(eval_fut)...)
    return res
end

end # module


