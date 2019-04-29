# Input matrix
function createrelmatrix(sentence)
    s_relation = Any[]
    for i in 1:length(sentence)
        for j in 1:length(sentence)
            push!(s_relation,vcat(sentence.cavec[i],sentence.cavec[j]))
        end
    end
    # pad_matrix = zeros(length(sentence.cavec[1])*2, (maxwordnum^2) - (length(sentence)^2))
    return hcat(s_relation...)
end

function predict(batches, w)
    relmatrices = Any[]
    for b in batches
        bpreds = Any[]
       for s in b
            push!(bpreds, non_lin(w, createrelmatrix(s)))   
        end
        push!(relmatrices, bpreds)
    end
    return relmatrices
end

# Not used anymore
function createygold(s)
   ygold = Any[]
    
    for i in 1:length(s)
        for j in 1:length(s)
            if(j == s.head[i])
                push!(ygold,1)
            else
                push!(ygold,0)
            end
        end
    end
    return transpose(ygold)
end


# integer based version
function collectygold(batches)
    ygold = Any[]
    for b in batches
        bgold = Any[]
       for s in b
            push!(bgold, s.head)
        end
        push!(ygold, bgold)
    end
    return ygold
end