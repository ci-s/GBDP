# Tools for improvement ideas

# Keep case, person kind of fields seperately/enumerated and train it in this way
uniq = String[]
for s in corpus
    for word_features in s.feats
       feats = split(word_features, "|")
        for f in feats
            field = split(f, "=")
            if !(field[1] in uniq)
                push!(uniq, field[1])
            end
        end 
    end
end