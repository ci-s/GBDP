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

# Creates a matrix that contains 1 for 
function ygoldroot(sentence)
        s=length(sentence)
        lambda_gold=zeros(s+1,s+1)
        for j=1:s
            lambda_gold[sentence.head[j]+1,j+1]=1.0
        end
    return lambda_gold
end


# Takes ypred as input and returns the extended version of it with 1.0 prediction for the root in the first row
function ypredroot(ypred,heads) # heads = s.head, ypred n x n matrix
    padded = vcat(KnetArray(map(Float32,zeros(1,length(heads)+1))),hcat(KnetArray(map(Float32,zeros(length(heads),1))),ypred))
    index = findall(i->i == 0, heads)[1] # ASSUMES ROOT ARC(ZERO) NO MORE THAN ONE TIME
    padded[1,index+1] = 1
    return padded
end

# Takes out the zero index
function mynll(y,a::AbstractArray{<:Integer}; dims=1, average=true)
    indices = Knet.findindices(y,a,dims=dims)
    for i in 1:length(a)
       if (a[i]==0)
           deleteat!(indices, i) 
            break
        end
    end
    lp = logp(y,dims=dims)[indices]
    average ? -mean(lp) : -sum(lp)
end


function sgdupdate!(func, args; lr=0.005)
    fval = @diff func(args...)
    for param in Knet.params(fval)
        ∇param = grad(fval, param)
        param .-= lr * ∇param
    end
    return value(fval)
end



function Eisner(n, lambda)
    #n=length(S)
    #E=Array{Any}(n,n,2,2)
    #=for s=1:n
        for d=1:2 c=1:2
            E[s,s,d,c]=0
        end
    end=#
    E=zeros(n+1,n+1,2,2)
    A=fill(Set(),(n+1,n+1,2,2))
    #A=Array{Any}(n,n,2,2)
    #temp_scores=zeros(2,2)
    #temp_sets=fill(Set(),(2,2))
    for m=1:n+1
        for s=1:n+1
            t=s+m
            t>n+1 && break
            max_score=max_q=-1
            for q=s:t-1
                tree_score=E[s,q,2,1]+E[q+1,t,1,1]+lambda[t,s] # line 20
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[1,2]=max_score
            E[s,t,1,2]=max_score
            if max_q>0
                #temp_sets[1,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(t,s)]))
                A[s,t,1,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(t-1,s-1)]))
            #else 
                #A[s,t,1,2]=union(A[s,s,2,1],A[s+1,t,1,1])
            end
            max_score=max_q=-1
            for q=s:t-1
                tree_score=E[s,q,2,1]+E[q+1,t,1,1]+lambda[s,t]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[2,2]=max_score
            E[s,t,2,2]=max_score
            if max_q>0
                #temp_sets[2,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(s,t)]))
                A[s,t,2,2]=union(A[s,max_q,2,1],A[max_q+1,t,1,1],Set([(s-1,t-1)]))
            #else
                #A[s,t,2,2]=union(A[s,s,2,1],A[s+1,t,1,1])
            end
            max_score=max_q=-1
            for q=s:t-1
                tree_score=E[s,q,1,1]+E[q,t,1,2]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[1,1]=max_score
            E[s,t,1,1]=max_score
            if max_q>0
                #temp_sets[1,1]=union(A[s,max_q,1,1],A[max_q,t,1,2])
                A[s,t,1,1]=union(A[s,max_q,1,1],A[max_q,t,1,2])
            end
            max_score=max_q=-1
            for q=s+1:t
                tree_score=E[s,q,2,2]+E[q,t,2,1]
                if tree_score>max_score
                    max_score=tree_score
                    max_q=q
                end
            end
            #temp_scores[2,1]=max_score
            E[s,t,2,1]=max_score
            if max_q>0
                #temp_sets[2,1]=union(A[s,max_q,2,2],A[max_q,t,2,1])
                A[s,t,2,1]=union(A[s,max_q,2,2],A[max_q,t,2,1])
            end
            #=for i=1:2 j=1:2
               E[s,t,i,j]=temp_scores[i,j]
               A[s,t,i,j]=temp_sets[i,j]
            end=#
        end
    end
    #=dependents=ones(n)
    for pair in A[1,n,2,1]
        dependents[pair[2]]=0
    end
    push!(A[1,n,2,1],(0,findfirst(dependents)))=#
    E[1,n+1,2,1],A[1,n+1,2,1]
end


function eisner(lambda)
    headsize = size(lambda,2)-1
    heads = Array{Int}(undef,headsize)
    score, arcset = Eisner(size(lambda,2)-1, lambda)
    for arc in arcset
        heads[arc[2]]=arc[1]
    end
    heads
end


function findmaxscores(ypred) # n+1 x n+1 scores matrix
    tuples = map(Tuple,argmax(Array(ypred), dims=1))
    indices = []
    for i in tuples
        push!(indices,i[1]) 
    end
    return indices
end


function createdeciderinput(s, maxheads, eisnerheads, deprels) # s = sentence, maxheads = argmax of predicted scores, eisnerheads = output of eisner algo
    decinput = []
    for i in 1:length(s)
        for j in 1:size(deprels,2)
            col = []
            push!(col,s.cavec[i])
            if (maxheads[i] == 0)
                push!(col, KnetArray(map(Float32,zeros(length(s.cavec[1])))))
            else
                push!(col,s.cavec[maxheads[i]])
            end

            if (eisnerheads[i] == 0)
                push!(col, KnetArray(map(Float32,zeros(length(s.cavec[1])))))
            else
                push!(col,s.cavec[eisnerheads[i]])
            end

            push!(col, KnetArray(map(Float32, deprels[:,j])))
            
            push!(decinput,vcat(col...))
            
        end
    end
    return decinput = hcat(decinput...) # [950*3 + 3200(deprel embedding size)] x [37 * Number of words in the sentence]
end