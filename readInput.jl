# Reads the conllu file as string
using DelimitedFiles

function readInput(filename)
    open(filename) do file
        corpus = Any[]
        sentence = Any[]
        for ln in eachline(file)
            if ln == ""
                push!(corpus,sentence)
                println(sentence," -> sentence pushed! \n")
                sentence = Any[]
            else
                push!(sentence,ln)
            end
        end
        return corpus
    end
end
