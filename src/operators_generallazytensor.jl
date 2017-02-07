"""
Lazy implementation of a tensor product of operators.
"""
type LazyTensor <: Operator
    N::Int
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    factor::Complex128
    operators::Dict{Vector{Int}, Operator}

    function LazyTensor(basis_l::Basis, basis_r::Basis,
                        operators::Dict{Vector{Int},Operator}, factor::Number=1)
        if typeof(basis_l) != CompositeBasis
            basis_l = CompositeBasis(basis_l)
        end
        if typeof(basis_r) != CompositeBasis
            basis_r = CompositeBasis(basis_r)
        end
        N = length(basis_l.bases)
        @assert N==length(basis_r.bases)
        for (I1, I2) in combinations(collect(keys(operators)), 2)
            if length(I1 ∩ I2) != 0
                throw(ArgumentError("Operators can't belong to common subsystems."))
            end
        end
        for (I, op) in operators
            @assert length(I) > 0
            if length(I) == 1
                @assert basis_l.bases[I[1]] == op.basis_l
                @assert basis_r.bases[I[1]] == op.basis_r
            else
                @assert basis_l.bases[I] == op.basis_l.bases
                @assert basis_r.bases[I] == op.basis_r.bases
            end
        end
        new(N, basis_l, basis_r, complex(factor), operators)
    end
end
LazyTensor{T}(basis_l::Basis, basis_r::Basis, operators::Dict{Vector{Int}, T}, factor::Number=1.) = LazyTensor(basis_l, basis_r, Dict{Vector{Int}, Operator}(item for item in operators), factor)
LazyTensor{T}(basis_l::Basis, basis_r::Basis, operators::Dict{Int, T}, factor::Number=1.) = LazyTensor(basis_l, basis_r, Dict{Vector{Int}, Operator}([i]=>op_i for (i, op_i) in operators), factor)
LazyTensor(basis::Basis, operators::Dict, factor::Number=1.) = LazyTensor(basis, basis, operators, factor)


function Base.full(op::LazyTensor)
    D = Dict{Vector{Int}, DenseOperator}(I=>full(op_I) for (I, op_I) in op.operators)
    for i in operators.complement(op.N, [keys(D)...;])
        D[[i]] = identityoperator(DenseOperator, op.basis_l.bases[i], op.basis_r.bases[i])
    end
    op.factor*embed(op.basis_l, op.basis_r, D)
end

function Base.sparse(op::LazyTensor)
    D = Dict{Vector{Int}, SparseOperator}(I=>sparse(op_I) for (I, op_I) in op.operators)
    for i in operators.complement(op.N, [keys(D)...;])
        D[[i]] = identityoperator(SparseOperator, op.basis_l.bases[i], op.basis_r.bases[i])
    end
    op.factor*embed(op.basis_l, op.basis_r, D)
end

==(x::LazyTensor, y::LazyTensor) = (x.basis_l == y.basis_l) && (x.basis_r == y.basis_r) && x.operators==y.operators && x.factor==y.factor

# Arithmetic
function *(a::LazyTensor, b::LazyTensor)
    check_multiplicable(a.basis_r, b.basis_l)
    N = a.N
    S = Set{Int}[]
    for I in sort([Set(x) for x in (keys(a.operators) ∪ keys(b.operators))], by=length, rev=true)
        if !any(I ⊆ s for s in S)
            @assert all(isempty(I ∩ s) for s in S)
            push!(S, I)
        end
    end
    D = Dict{Vector{Int}, Operator}()
    for s in S
        # Find operators that are in that subspace
        D_s1 = Dict(I=>op_I for (I, op_I) in a.operators if I ⊆ s)
        D_s2 = Dict(I=>op_I for (I, op_I) in b.operators if I ⊆ s)
        if length(D_s1) == 0
            @assert length(D_s2) == 1
            I, op_i = first(D_s2)
            D[I] = op_i
        elseif length(D_s2) == 0
            @assert length(D_s1) == 1
            I, op_i = first(D_s1)
            D[I] = op_i
        else
            # Indices of subsystems are sorted in the resulting operator
            I = sort(collect(s))
            # Squash down indices
            I_ = operators.complement(N, I)
            D_s1 = removeindices(D_s1, I_)
            D_s2 = removeindices(D_s2, I_)
            a_I = embed(ptrace(a.basis_l, I_), ptrace(a.basis_r, I_), D_s1)
            b_I = embed(ptrace(b.basis_l, I_), ptrace(b.basis_r, I_), D_s2)
            D[I] = a_I*b_I
        end
    end
    LazyTensor(a.basis_l, b.basis_r, D, a.factor*b.factor)
end
*(a::LazyTensor, b::Number) = LazyTensor(a.basis_l, a.basis_r, a.operators, a.factor*b)
*(a::Number, b::LazyTensor) = LazyTensor(b.basis_l, b.basis_r, b.operators, a*b.factor)

/(a::LazyTensor, b::Number) = LazyTensor(a.basis_l, a.basis_r, a.operators, a.factor/b)

-(a::LazyTensor) = LazyTensor(a.basis_l, a.basis_r, a.operators, -a.factor)


identityoperator(::Type{LazyTensor}, b1::Basis, b2::Basis) = LazyTensor(b1, b2, Dict{Vector{Int},Operator}())

dagger(op::LazyTensor) = LazyTensor(op.basis_r, op.basis_l, Dict(I=>dagger(op_I) for (I, op_I) in op.operators), conj(op.factor))

_identitylength(op::LazyTensor, i::Int) = min(length(op.basis_l.bases[i]), length(op.basis_r.bases[i]))
_identitytrace(op::LazyTensor, indices::Vector{Int}) = prod(Complex128[_identitylength(op, i) for i in complementindices(op.N, op.operators) ∩ indices])
_identitytrace(op::LazyTensor) = prod(Complex128[_identitylength(op, i) for i in complementindices(op.N, op.operators)])

trace(op::LazyTensor) = op.factor*prod(Complex128[trace(op_I) for op_I in values(op.operators)])*_identitytrace(op)

function ptrace(op::LazyTensor, indices::Vector{Int})
    operators.check_ptrace_arguments(op, indices)
    rank = op.N - length(indices)
    if rank==0
        return trace(op)
    end
    D = Dict{Vector{Int},Operator}()
    factor = op.factor*_identitytrace(op, indices)
    for (I, op_I) in op.operators
        J = I ∩ indices
        if length(J) == length(I)
            factor *= trace(op_I)
        elseif length(J) == 0
            D[I] = op_I
        else
            subindices = [findfirst(I, j) for j in J]
            D[setdiff(I, indices)] = ptrace(op_I, subindices)
        end
    end
    if rank==1 && length(D)==1
        return factor*first(values(D))
    end
    b_l = ptrace(op.basis_l, indices)
    b_r = ptrace(op.basis_r, indices)
    if rank==1
        return identityoperator(b_l, b_r) * factor
    end
    D = removeindices(D, indices)
    LazyTensor(b_l, b_r, D, factor)
end

function removeindices{T}(D::Dict{Vector{Int},T}, indices::Vector{Int})
    result = Dict{Vector{Int},T}()
    for (I, op_I) in D
        @assert isempty(I ∩ indices)
        J = [i-sum(indices.<i) for i in I]
        result[J] = op_I
    end
    result
end
complementindices{T}(N::Int, D::Dict{Vector{Int},T}) = operators.complement(N, [keys(D)...;])
shiftindices{T}(D::Dict{Vector{Int},T}, offset::Int) = Dict{Vector{Int},T}(I+offset=>op_I for (I, op_I) in D)

tensor(a::LazyTensor, b::LazyTensor) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, merge(a.operators, shiftindices(b.operators, a.N)), a.factor*b.factor)
tensor(a::LazyWrapper, b::LazyWrapper) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, Dict([1]=>a.operator, [2]=>b.operator), a.factor*b.factor)
tensor(a::LazyTensor, b::LazyWrapper) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, merge(a.operators, Dict([a.N+1]=>b.operator)), a.factor*b.factor)
tensor(a::LazyWrapper, b::LazyTensor) = LazyTensor(a.basis_l ⊗ b.basis_l, a.basis_r ⊗ b.basis_r, merge(Dict([1]=>a.operator), shiftindices(b.operators, 1)), a.factor*b.factor)


function permutesystems(op::LazyTensor, perm::Vector{Int})
    b_l = permutesystems(op.basis_l, perm)
    b_r = permutesystems(op.basis_r, perm)
    operators = Dict{Vector{Int},Operator}([findfirst(perm, i) for i in I]=>op_I for (I, op_I) in op.operators)
    LazyTensor(b_l, b_r, operators, op.factor)
end
