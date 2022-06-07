@testset "evaluate model" begin

    batch_size=8

    @everywhere p evalDataChan = Channel(1) do ch
        n_chunk = ceil(Int,size(x_array)[1]/$batch_size)
        x_dat = Flux.chunk(transpose(x_array), n_chunk)
        for d in x_dat
            put!(ch, d)
        end
        
    end

    @everywhere p evalDatRemChan = RemoteChannel(() -> evalDataChan, myid())

    test_path = joinpath(splitpath(pathof(DistributedFluxML))[1:end-2]...,
                         "test")

    trainWorkers_shift = circshift(p, 1)
    # ^^^ shift workers to reuse workers as ^^^
    # ^^^ remote data hosts ^^^
    datRemChansDict = Dict(k => @fetchfrom w evalDatRemChan
                           for (k,w) in zip(p, trainWorkers_shift))

    model = @fetchfrom p[1] DistributedFluxML.model

    global res = DistributedFluxML.eval_model(model, datRemChansDict, p; status_chan)

    y = vcat([@fetchfrom w Flux.chunk(y_array, ceil(Int,size(y_array)[2]/batch_size)) for w in trainWorkers_shift]...)

    loss_f = Flux.Losses.logitcrossentropy

    evalLosses = [loss_f(r,_y) for (r, _y) in zip(res, y)]
    n_steps_in_batch = length(evalLosses)
    last_step = maximum([s[:step] for s in status_array
                         if (s[:statusName] == "do_train_on_remote.step.grad")])
    trainLosses = [s[:loss] for s in status_array
                   if (s[:statusName] == "do_train_on_remote.step.grad") &
                   (s[:step] > (last_step - n_steps_in_batch))]

    @test mean(evalLosses) < mean(trainLosses) + sqrt(r2(ols))*3 # test that eval_model used the model that we passed

    max_block = maximum([length(i) for i in y])
    shuf_mask = [length(i) == max_block for i in y]
    shuffle_idx = collect(1:(length(y[shuf_mask])))
    shuffle!(shuffle_idx)
    shufEvalLosses = [loss_f(r,_y) for (r, _y) in zip(res[shuf_mask], y[shuf_mask][shuffle_idx])]

    @test mean(shufEvalLosses) > mean(evalLosses) #test that eval_model consitantly agrigates w.r.t. it's input 
end
