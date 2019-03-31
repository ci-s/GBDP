# Create an array that contains all unique language specific part-of-speech-tags (xpostag)
uniq = String[]
for s in corpus
    for w in s.xpostag
        if !(w in uniq)
            push!(uniq, w)
        end
    end
end


# Give a number to each xpostag
xpostag = Dict{String, UInt8}()
for i in 1:length(uniq)
    xpostag[uniq[i]] = i
end