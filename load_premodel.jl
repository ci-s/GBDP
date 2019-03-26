
using JLD

d = JLD.load("pretrained_model.jld2")

include("types.jl")
include("pre_processing.jl")

v = create_vocab(d)

corpus = load_conllu("tr_imst-ud-train.conllu",v)

corpus[1]

typeof(corpus[1])

s = corpus[1]
s.word

s.postag

s.head

s.deprel

s.vocab


