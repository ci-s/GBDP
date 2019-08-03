
include("types.jl")
#import JLD2, Knet
#Pkg.add("JLD")
#Pkg.add("Knet")
using JLD2,Knet


const language_model = "english_chmodel.jld"
const data_file = "en-ud-dev.conllu"



function load_conllu(file,v::Vocab)
    corpus = Any[]
    s = Sentence(v)
    for line in eachline(file)
        if line == ""
            push!(corpus, s)
            s = Sentence(v)
        elseif (m = match(r"^\d+\t(.+?)\t.+?\t(.+?)\t.+?\t.+?\t(.+?)\t(.+?)(:.+)?\t", line)) != nothing # modify that to use different columns
            #                id   word   lem  upos   xpos feat head   deprel
            #println(m.match, summary(m.match))
            #println()
            match_str = split(String(m.match), "\t")
            push!(s.xpostag, String(match_str[5]))
            push!(s.feats, String(match_str[6]))
            
            word = m.captures[1]
            push!(s.word, word)
            postag = get(v.postags, m.captures[2], 0)
            if postag==0
                Base.warn_once("Unknown postags")
            end
            push!(s.postag, postag)
            
            head = tryparse(Position, m.captures[3])
            head = (head==nothing) ? -1 : head
            if head==-1
                Base.warn_once("Unknown heads")
            end
            push!(s.head, head)

            deprel = get(v.deprels, m.captures[4], 0)
            if deprel==0
                Base.warn_once("Unknown deprels")
            end
            push!(s.deprel, deprel)
        end
    end
    return corpus
end


# To create vocabulary from pre-trained lstm model, modify that to use different cols
function create_vocab(d)
    Vocab(d["char_vocab"],
          Dict{String, Int}(),
          d["word_vocab"],
          d["sosword"],
          d["eosword"],
          d["unkword"],
          d["sowchar"],
          d["eowchar"],
          d["unkchar"],
          get(d, "postags", UPOSTAG),
          get(d, "deprels", UDEPREL)
          )
end


function extend_vocab!(vocab::Vocab, train_corpus)
    xpos = Dict()
    feats = Dict()
    xpos["<unk>"] = 1
    feats["<unk>"] = 1
    for sent in train_corpus
        sent.xpostag = #[proc(xp, xpos) for xp in sent.xpostag]
            [get!(xpos, xp, 1+length(xpos)) for xp in sent.xpostag]
        sent.feats = [proc(feat, feats) for feat in sent.feats]
    end

    return ExtendedVocab(xpos, feats, vocab)
end


function extend_corpus!(ev::ExtendedVocab, val_corpus)
    for sent in val_corpus
        sent.xpostag = [get(ev.xpostags, xp, ev.xpostags["<unk>"]) 
                        for xp in sent.xpostag]
        sent.feats = [proc(f, ev.feats, false)
                      for f in sent.feats]
    end
    return val_corpus
end


function proc(feat, feats, mutate=true)
    feat == "_" && return []
    splits = split(feat, "|")
    stuff = []
    for s in splits
        push!(stuff, mutate ? get!(feats, s, length(feats)+1) : get(feats, s, feats["<unk>"]))
    end
    return stuff
end 


function minibatch(corpus, batchsize; maxlen=typemax(Int), minlen=1, shuf=false)
    data = Any[]
    sorted = sort(corpus, by=length)
    i1 = findfirst(x->(length(x) >= minlen), sorted)
    if i1==0; error("No sentences >= $minlen"); end
    i2 = findlast(x->(length(x) <= maxlen), sorted)
    if i2==0; error("No sentences <= $maxlen"); end
    for i in i1:batchsize:i2
        j = min(i2, i+batchsize-1)
        push!(data, sorted[i:j])
    end
    if shuf
        data=shuffle(data)
    end
    return data
end


# Things we added

function model_charfix(d)
    d["new_d2"]["unkchar"] = Char(0x11)
    d["new_d2"]["sowchar"] = Char(0x12)
    d["new_d2"]["eowchar"] = Char(0x13)
    return d["new_d2"]
end

