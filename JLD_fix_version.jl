
include("types.jl")
using Knet, JLD2, JLD
using DelimitedFiles

d = load("model.jld2")

words = readdlm("word_vocab.txt")
word_vocab = Dict{AbstractString, Int64}()

for i in 1:2:length(words)
    push!(word_vocab, string(words[i]) => words[i+1])
end

word_vocab

push!(d, "word_vocab" => word_vocab)

using FileIO
FileIO.save("pretrained_model.jld2", d)


