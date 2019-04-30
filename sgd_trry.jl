
using JLD,JLD2,Knet
using Base.Iterators: flatten
using Random, Distributions
Random.seed!(5)

include("types.jl")
include("pre_processing.jl")
include("encoder_utils.jl")
include("decoder_utils.jl")

d = JLD.load("english_chmodel3.jld2")
d["new_d2"]["unkchar"] = Char(0x11)
d["new_d2"]["sowchar"] = Char(0x12)
d["new_d2"]["eowchar"] = Char(0x13)

v = create_vocab(d["new_d2"])
corpus = load_conllu("tr_imst-ud-train.conllu",v)
wmodel = makewmodel(d["new_d2"])

s = corpus[1]

ev = extend_vocab!(v, corpus)
fs = createcolvecs(corpus,ev)

fillvecs!(wmodel, corpus, v, fs)

fillcavec!(corpus)


function non_lin(w,x)
   return transpose(w[1:length(w)-1])*x .+ w[length(w)] 
end

# Going into autograd

# ******
struct Linear; w; b; end
(m::Linear)(x) = m.w * x .+ m.b
(m::Linear)(x, y) = Knet.nll(reshape(m(x), (length(y), length(y))), y)
#(m::Linear)(data::Data) = mean(m(x,y) for (x,y) in data)
Linear(i::Int,o::Int,scale=0.01) = Linear(Param(KnetArray(map(Float32,scale * randn(o,i)))), Param(KnetArray(map(Float32,zeros(o)))))

# ***
model = Linear(length(corpus[1].cavec[1]) * 2, 1)

function sgdupdate!(func, args; lr=0.1)
    fval = @diff func(args...)
    for param in Knet.params(fval)
        ∇param = grad(fval, param)
        param .-= lr * ∇param
    end
    return value(fval)
end

bitches = minibatch(corpus, 10)

#for b in bitches[80:150]
   for s in corpus #b
        if length(s)>2
            println(s.word)
            rm = createrelmatrix(s)
            println(size(rm))
            loss = sgdupdate!(model, (rm, s.head))
            println(loss) 
        end
    end 
    #sleep(10)
#end
