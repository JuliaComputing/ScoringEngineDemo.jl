using DataFrames
using Statistics: mean, std

abstract type Projector <: Function end

struct Normalizer <: Projector
    μ
    σ
end

struct Quantilizer <: Projector
    quantiles
end

Normalizer(x::AbstractVector) = Normalizer(mean(x), std(x))

function (m::Normalizer)(x::Real)
    return (x - m.μ) / m.σ
end

function (m::Normalizer)(x::AbstractVector)
    return (x .- m.μ) ./ m.σ
end

df = DataFrame(:v1 => rand(5), :v2 => rand(5))
feat_names = names(df)
norms = map((feat) -> Normalizer(df[:, feat]), feat_names)

# ByRow works
transform(df, feat_names .=> ByRow.(norms) .=> feat_names)

# Does not work
transform(df, feat_names[1] => norms[1] => feat_names[1])
# Does work
transform(df, feat_names[1] => (x -> norms[1](x)) => feat_names[1])

# Does not work
transform(df, feat_names .=> norms .=> feat_names)

# Does work
norms_f = map(f -> (x) -> f(x), norms)
transform(df, feat_names .=> norms_f .=> feat_names)



#################
# multi-function pattern
using DataFrames
using BenchmarkTools
nrows = 1_000_000
df = DataFrame(id=rand(["A"], nrows), v1=rand(nrows), v2=rand(nrows), v3=rand(nrows), v4=rand(nrows))
df = DataFrame(id=rand(["A", "B", "C", "D", "E", "F", "G", "H" ,"I", "J"], nrows), v1=rand(nrows), v2=rand(nrows), v3=rand(nrows), v4=rand(nrows))

# fun1(x) = exp(x)
# fun2(x) = exp(x)
# fun3(x) = exp(x)
# fun4(x) = exp(x)

f1 = "v1" => ByRow(exp) => "new1"
f2 = "v2" => ByRow(exp) => "new2"
f3 = "v2" => ByRow(exp) => "new3"
f4 = "v2" => ByRow(exp) => "new4"

funs = [f1, f2, f3, f4]

function df_trans_A(df, funs)
    transform!(df, funs[1])
    transform!(df, funs[2])
    transform!(df, funs[3])
    transform!(df, funs[4])
end

function df_trans_B(df, funs)
    transform!(df, funs)
end

# 19.500 ms (644 allocations: 30.55 MiB)
@btime df_trans_A($df, $funs);
# 19.325 ms (410 allocations: 30.54 MiB)
@btime df_trans_B($df, $funs);

gdf = groupby(df, "id")
function gdf_trans_A(gdf, funs)
    transform!(gdf, funs[1])
    transform!(gdf, funs[2])
    transform!(gdf, funs[3])
    transform!(gdf, funs[4])
end

function gdf_trans_B(gdf, funs)
    transform!(gdf, funs)
end

@time gdf_trans_A(gdf, funs);
@time gdf_trans_B(gdf, funs);

# Single group
# 56.028 ms (2963 allocations: 152.76 MiB)
@btime gdf_trans_A($gdf, $funs);
# 19.652 ms (1124 allocations: 129.76 MiB)
@btime gdf_trans_B($gdf, $funs);

# 10 groups
# 114.379 ms (3284 allocations: 195.40 MiB)
@btime gdf_trans_A($gdf, $funs);
# 49.737 ms (1444 allocations: 172.41 MiB)
@btime gdf_trans_B($gdf, $funs);



using DataFrames
using BenchmarkTools
nrows = 1_000_000
df = DataFrame(id=rand(["A", "B", "C", "D", "E", "F", "G", "H" ,"I", "J"], nrows), v1=rand(nrows), v2=rand(nrows), v3=rand(nrows), v4=rand(nrows))
df = DataFrame(id=rand(["A"], nrows), v1=rand(nrows), v2=rand(nrows), v3=rand(nrows), v4=rand(nrows))

f1 = "v1" => sum => "new1"
f2 = "v2" => sum => "new2"
f3 = "v2" => sum => "new3"
f4 = "v2" => sum => "new4"

funs = [f1, f2, f3, f4]

function df_trans_A(df, funs)
    dfg = groupby(df, :id)
    agg = combine(dfg, funs[1])
    agg = combine(dfg, funs[2])
    agg = combine(dfg, funs[3])
    agg = combine(dfg, funs[4])
end

function df_trans_B(df, funs)
    dfg = groupby(df, :id)
    agg = combine(dfg, funs)
end

# 1 Group
# 26.565 ms (1067 allocations: 31.33 MiB)
@btime df_trans_A($df, $funs);
# 17.194 ms (546 allocations: 31.29 MiB)
@btime df_trans_B($df, $funs);

# 10 Groups
# 18.536 ms (1068 allocations: 31.33 MiB)
@btime df_trans_A($df, $funs);
# 15.977 ms (546 allocations: 31.29 MiB)
@btime df_trans_B($df, $funs);

function df_trans_A(df, funs)
    agg = combine(df, funs[1])
    agg = combine(df, funs[2])
    agg = combine(df, funs[3])
    agg = combine(df, funs[4])
end

function df_trans_B(df, funs)
    agg = combine(df, funs)
end

# 1.664 ms (444 allocations: 28.44 KiB)
@btime df_trans_A($df, $funs);
# 1.640 ms (382 allocations: 21.69 KiB)
@btime df_trans_B($df, $funs);

# 1.665 ms (444 allocations: 28.44 KiB)
@btime df_trans_A($df, $funs);
# 1.631 ms (382 allocations: 21.69 KiB)
@btime df_trans_B($df, $funs);