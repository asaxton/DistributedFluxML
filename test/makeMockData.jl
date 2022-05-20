using DataFrames
using CSV
using Random
using Serialization

mockData_path = joinpath(splitpath(pathof(DistributedFluxML))[1:end-2]...,"mockData")
data_fn = joinpath(mockData_path, "iris.data") #bezdekIris.data

col_names = ["sepal_l", "sepal_w", "petal_l", "petal_w", "class"]
df_tot = CSV.read(data_fn, DataFrame; header=col_names)

data_shard_fns = [joinpath(mockData_path, "iris_df_$(i).jlb") for i in 1:3]

sh_idx = collect(1:nrow(df_tot))
shuffle!(sh_idx)

for (fn, idx) in zip(data_shard_fns, [sh_idx[1:50], sh_idx[51:100], sh_idx[101:end]])
    serialize(fn, df_tot[idx, :])
end
