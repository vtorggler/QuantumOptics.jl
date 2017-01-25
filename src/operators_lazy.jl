module operators_lazy

import Base: ==, *, /, +, -
import ..operators: dagger, identityoperator,
                    trace, ptrace, normalize!, tensor, permutesystems

using Combinatorics
using Base.Cartesian
using ..bases, ..states, ..operators, ..operators_dense, ..operators_sparse

export LazyOperator, LazyTensor, LazySum, LazyProduct


"""
Abstract base class for lazy operators.

Lazy means in this context that operator-operator operations are not performed
immediately but delayed until the operator is applied to a state, e.g. instead
of (A + B) * x it calculates (A * x) + (B * x).

Lazy operations are implemented for:
    * TensorProduct: LazyTensor
    * Addition: LazySum
    * Multiplication: LazyProduct
"""
abstract LazyOperator <: Operator


"""
Lazy implementation of a tensor product of operators.

The suboperators are stored as values in a dictionary where the key is
the index of the subsystem. Additionally a complex factor is stored in the
"factor" field which allows for fast multiplication with a number.
"""
type LazyTensor <: LazyOperator
    N::Int
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    factor::Complex128
    operators::Dict{Vector{Int}, Operator}

    function LazyTensor(basis_l::CompositeBasis, basis_r::CompositeBasis,
                        operators::Dict{Vector{Int},Operator}, factor::Number=1.)
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
LazyTensor{T}(basis_l::CompositeBasis, basis_r::CompositeBasis, operators::Dict{Vector{Int}, T}, factor::Number=1.) = LazyTensor(basis_l, basis_r, Dict{Vector{Int}, Operator}(item for item in operators), factor)
LazyTensor{T}(basis_l::CompositeBasis, basis_r::CompositeBasis, operators::Dict{Int, T}, factor::Number=1.) = LazyTensor(basis_l, basis_r, Dict{Vector{Int}, Operator}([i]=>op_i for (i, op_i) in operators), factor)
LazyTensor(basis::CompositeBasis, operators::Dict, factor::Number=1.) = LazyTensor(basis, basis, operators, factor)


"""
Lazy evaluation of sum of operators.

All operators have to be given in respect to the same bases. The field
factors accounts for an additional multiplicative factor for each operator.
"""
type LazySum <: LazyOperator
    basis_l::Basis
    basis_r::Basis
    factors::Vector{Complex128}
    operators::Vector{Operator}

    function LazySum(factors::Vector{Complex128}, operators::Vector{Operator})
        @assert length(operators)>0
        @assert length(operators)==length(factors)
        for i = 2:length(operators)
            @assert operators[1].basis_l == operators[i].basis_l
            @assert operators[1].basis_r == operators[i].basis_r
        end
        new(operators[1].basis_l, operators[1].basis_r, factors, operators)
    end
end
LazySum(operators::Operator...) = LazySum(ones(Complex128, length(operators)), Operator[operators...])


"""
Lazy evaluation of product of operators.

The factors of the product are stored in the "operators" field. Additionally a
complex factor is stored in the "factor" field which allows for fast
multiplication with a number.
"""
type LazyProduct <: LazyOperator
    basis_l::Basis
    basis_r::Basis
    factor::Complex128
    operators::Vector{Operator}

    function LazyProduct(factor::Complex128, operators::Vector{Operator})
        for i = 2:length(operators)
            @assert multiplicable(operators[i-1].basis_r, operators[i].basis_l)
        end
        new(operators[1].basis_l, operators[end].basis_r, factor, operators)
    end
end
LazyProduct(operators::Operator...) = LazyProduct(Complex(1.), Operator[operators...])

function Base.full(op::LazyTensor)
    D = Dict{Vector{Int}, DenseOperator}(I=>full(op_I) for (I, op_I) in op.operators)
    for i in operators.complement(op.N, [keys(D)...;])
        D[[i]] = identityoperator(DenseOperator, op.basis_l.bases[i], op.basis_r.bases[i])
    end
    op.factor*embed(op.basis_l, op.basis_r, D)
end
Base.full(op::LazySum) = sum(a*full(op_i) for (a, op_i) in zip(op.factors, op.operators))
Base.full(op::LazyProduct) = op.factor*prod(full(op_i) for op_i in op.operators)

function Base.sparse(op::LazyTensor)
    D = Dict{Vector{Int}, SparseOperator}(I=>sparse(op_I) for (I, op_I) in op.operators)
    for i in operators.complement(op.N, [keys(D)...;])
        D[[i]] = identityoperator(SparseOperator, op.basis_l.bases[i], op.basis_r.bases[i])
    end
    op.factor*embed(op.basis_l, op.basis_r, D)
end
Base.sparse(op::LazySum) = sum(a*sparse(op_i) for (a, op_i) in zip(op.factors, op.operators))
Base.sparse(op::LazyProduct) = op.factor*prod(sparse(op_i) for op_i in op.operators)


# Arithmetic
function matchfactors(D_l::Dict{Vector{Int}, Operator}, D_r::Dict{Vector{Int}, Operator})
    D::Dict{Vector{Int}, (Vector{Operator}, Vector{Operator})}
    for (I, op_I) in D
    end
end

function *(a::LazyTensor, b::LazyTensor)
    check_multiplicable(a.basis_r, b.basis_l)
    a_indices = keys(a.operators)
    b_indices = keys(b.operators)
    D = Dict{Int, Operator}()
    for i=intersect(a_indices, b_indices)
        D[i] = a.operators[i]*b.operators[i]
    end
    for i=setdiff(a_indices, b_indices)
        D[i] = a.operators[i]
    end
    for i=setdiff(b_indices, a_indices)
        D[i] = b.operators[i]
    end
    return LazyTensor(a.basis_l, b.basis_r, D, a.factor*b.factor)
end
*(a::LazyTensor, b::Number) = LazyTensor(a.basis_l, a.basis_r, a.operators, a.factor*b)
*(a::Number, b::LazyTensor) = LazyTensor(b.basis_l, b.basis_r, b.operators, a*b.factor)

# *(a::LazySum, b::LazySum) = LazySum([op1*op2 for op1=a.operators, op2=b.operators]...)
*(a::LazySum, b::LazySum) = LazyProduct(a, b)
*(a::LazySum, b::Number) = LazySum(b*a.factors, a.operators)
*(a::Number, b::LazySum) = LazySum(a*b.factors, b.operators)

*(a::LazyProduct, b::LazyProduct) = LazyProduct(a.factor*b.factor, a.operators..., b.operators...)
*(a::LazyProduct, b::Number) = LazyProduct(a.factor*b, a.operators...)
*(a::Number, b::LazyProduct) = LazyProduct(a*b.factor, b.operators...)

+(a::LazyOperator, b::LazyOperator) = LazySum(a, b)
+(a::LazyOperator, b::Operator) = LazySum(a, b)
+(a::Operator, b::LazyOperator) = LazySum(a, b)
+(a::LazySum, b::LazySum) = LazySum([a.factors, b.factors;], [a.operators, b.operators;])
+(a::LazySum, b::LazyOperator) = LazySum([a.factors, Complex(1.);], [a.operators, b;])
+(a::LazyOperator, b::LazySum) = LazySum([Complex(1.), b.factors;], [a, b.operators;])
+(a::LazySum, b::Operator) = LazySum([a.factors, Complex(1.);], [a.operators, b;])
+(a::Operator, b::LazySum) = LazySum([a.factors, Complex(1.);], [a, b.operators;])

-(a::LazyTensor) = LazyTensor(a.basis_l, a.basis_r, a.operators, -a.factor)
-(a::LazySum) = LazySum(-a.factors, a.operators)
-(a::LazyProduct) = LazyProduct(-a.factor, a.operators)

-(a::LazyOperator, b::LazyOperator) = LazySum([Complex(1.), Complex(-1.)], [a, b])
-(a::LazyOperator, b::Operator) = LazySum([Complex(1.), Complex(-1.)], [a, b])
-(a::Operator, b::LazyOperator) = LazySum([Complex(1.), Complex(-1.)], [a, b])
-(a::LazySum, b::LazySum) = LazySum([a.factors, -b.factors], [a.operators, b.operators])
-(a::LazySum, b::LazyOperator) = LazySum([a.factors, -Complex(1.)], [a.operators, b])
-(a::LazyOperator, b::LazySum) = LazySum([Complex(1.), -a.factors], [a, b.operators])
-(a::LazySum, b::Operator) = LazySum([a.factors, -Complex(1.)], [a.operators, b])
-(a::Operator, b::LazySum) = LazySum([Complex(1.), -a.factors], [a, b.operators])


identityoperator(::Type{LazyTensor}, b1::Basis, b2::Basis) = LazyTensor(b1, b2, Dict{Vector{Int},Operator}())
identityoperator(::Type{LazySum}, b1::Basis, b2::Basis) = LazySum(identityoperator(b1, b2))
identityoperator(::Type{LazyProduct}, b1::Basis, b2::Basis) = LazyProduct(identityoperator(b1, b2))

dagger(op::LazyTensor) = LazyTensor(op.basis_r, op.basis_l, Dict(I=>dagger(op_I) for (I, op_I) in op.operators), conj(op.factor))
dagger(op::LazySum) = LazySum(conj(op.factors), Operator[dagger(op_i) for op_i in op.operators])
dagger(op::LazyProduct) = LazyProduct(conj(op.factor), [dagger(op_i) for op_i in reverse(op.operators)])

trace(op::LazyTensor) = op.factor*prod([(haskey(op.operators,i) ? trace(op.operators[i]): prod(op.basis_l.shape)) for i=1:length(op.basis_l.bases)])
trace(op::LazySum) = sum([trace(x) for x=op.operators])
trace(op::LazyProduct) = throw(ArgumentError("Trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. full() or sparse()."))

function ptrace(op::LazyTensor, indices::Vector{Int})
    operators.check_ptrace_arguments(op, indices)
    rank = length(op.basis_l.shape) - length(indices)
    if rank==0
        return trace(op)
    end
    D = Dict{Int,Operator}()
    factor = op.factor
    for (i, op_i) in op.operators
        if i in indices
            factor *= trace(op_i)
        else
            D[i] = op_i
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
    LazyTensor(b_l, b_r, D, factor)
end

function ptrace(op::LazySum, indices::Vector{Int})
    operators.check_ptrace_arguments(op, indices)
    rank = length(op.basis_l.shape) - length(indices)
    if rank==0
        return trace(op)
    end
    D = Operator[ptrace(op_i, indices) for op_i in op.operators]
    LazySum(op.factors, D)
end

ptrace(op::LazyProduct, indices::Vector{Int}) = throw(ArgumentError("Trace of LazyProduct is not defined. Try to convert to another operator type first with e.g. full() or sparse()."))


function permutesystems(op::LazyTensor, perm::Vector{Int})
    b_l = permutesystems(op.basis_l, perm)
    b_r = permutesystems(op.basis_r, perm)
    operators = Dict{Int,Operator}(findfirst(perm, i)=>op_i for (i, op_i) in op.operators)
    LazyTensor(b_l, b_r, operators, op.factor)
end
operators.permutesystems(op::LazySum, perm::Vector{Int}) = LazySum(op.factors, Operator[permutesystems(op_i, perm) for op_i in op.operators])
operators.permutesystems(op::LazyProduct, perm::Vector{Int}) = LazyProduct(op.factor, Operator[permutesystems(op_i, perm) for op_i in op.operators])


@generated function _lazytensor_gemv!{RANK, INDEX}(rank::Array{Int, RANK}, index::Array{Int, INDEX},
                                        alpha, op::Operator, b::Ket, beta, result::Ket)
    return quote
        x = Ket(b.basis.bases[$INDEX])
        y = Ket(result.basis.bases[$INDEX])
        indices_others = filter(x->(x!=$INDEX), 1:$RANK)
        shape_others = [b.basis.shape[i] for i=indices_others]
        strides_others = [operators_dense._strides(b.basis.shape)[i] for i=indices_others]
        stride = operators_dense._strides(b.basis.shape)[$INDEX]
        N = b.basis.shape[$INDEX]
        @nexprs 1 d->(I_{$(RANK-1)}=1)
        @nloops $(RANK-1) i d->(1:shape_others[d]) d->(I_{d-1}=I_{d}) d->(I_d+=strides_others[d]) begin
            for j=1:N
                x.data[j] = b.data[I_0+stride*(j-1)]
            end
            operators.gemv!(alpha, op, x, beta, y)
            for j=1:N
                result.data[I_0+stride*(j-1)] = y.data[j]
            end
        end
    end
end

function operators.gemv!(alpha, a::LazyTensor, b::Ket, beta, result::Ket)
    rank = zeros(Int, [0 for i=1:length(a.basis_l.shape)]...)
    bases = [b for b=b.basis.bases]
    for op_index in keys(a.operators)
        index = zeros(Int, [0 for i=1:op_index]...)
        bases[op_index] = a.operators[op_index].basis_l
        tmp = Ket(CompositeBasis(bases...))
        _lazytensor_gemv!(rank, index, Complex(1.), a.operators[op_index], b, Complex(0.), tmp)
        b = tmp
    end
    result.data[:] = beta*result.data[:] + b.data[:]
    nothing
end

@generated function _lazytensor_gemv!{RANK, INDEX}(rank::Array{Int, RANK}, index::Array{Int, INDEX},
                                        alpha, b::Bra, op::Operator, beta, result::Bra)
    return quote
        x = Bra(b.basis.bases[$INDEX])
        y = Bra(result.basis.bases[$INDEX])
        indices_others = filter(x->(x!=$INDEX), 1:$RANK)
        shape_others = [b.basis.shape[i] for i=indices_others]
        strides_others = [operators_dense._strides(b.basis.shape)[i] for i=indices_others]
        stride = operators_dense._strides(b.basis.shape)[$INDEX]
        N = b.basis.shape[$INDEX]
        @nexprs 1 d->(I_{$(RANK-1)}=1)
        @nloops $(RANK-1) i d->(1:shape_others[d]) d->(I_{d-1}=I_{d}) d->(I_d+=strides_others[d]) begin
            for j=1:N
                x.data[j] = b.data[I_0+stride*(j-1)]
            end
            operators.gemv!(alpha, x, op, beta, y)
            for j=1:N
                result.data[I_0+stride*(j-1)] = y.data[j]
            end
        end
    end
end

function operators.gemv!(alpha, a::Bra, b::LazyTensor, beta, result::Bra)
    rank = zeros(Int, [0 for i=1:length(b.basis_r.shape)]...)
    bases = [b for b=a.basis.bases]
    for op_index in keys(b.operators)
        index = zeros(Int, [0 for i=1:op_index]...)
        bases[op_index] = b.operators[op_index].basis_r
        tmp = Bra(CompositeBasis(bases...))
        _lazytensor_gemv!(rank, index, Complex(1.), a, b.operators[op_index], Complex(0.), tmp)
        a = tmp
    end
    result.data[:] = beta*result.data[:] + a.data[:]
    nothing
end

function operators.gemv!(alpha, a::LazySum, b::Ket, beta, result::Ket)
    operators.gemv!(alpha*a.factors[1], a.operators[1], b, beta, result)
    for i=2:length(a.operators)
        operators.gemv!(alpha*a.factors[i], a.operators[i], b, Complex(1.), result)
    end
end

function operators.gemv!(alpha, a::Bra, b::LazySum, beta, result::Bra)
    operators.gemv!(alpha*b.factors[1], a, b.operators[1], beta, result)
    for i=2:length(b.operators)
        operators.gemv!(alpha*b.factors[i], a, b.operators[i], Complex(1.), result)
    end
end

function operators.gemv!(alpha, a::LazyProduct, b::Ket, beta, result::Ket)
    tmp1 = Ket(a.operators[end].basis_l)
    operators.gemv!(Complex(1.)*a.factor, a.operators[end], b, Complex(0.), tmp1)
    for i=length(a.operators)-1:-1:2
        tmp2 = Ket(a.operators[i].basis_l)
        operators.gemv!(Complex(1.), a.operators[i], tmp1, Complex(0.), tmp2)
        tmp1 = tmp2
    end
    operators.gemv!(alpha, a.operators[1], tmp1, beta, result)
end

function operators.gemv!(alpha, a::Bra, b::LazyProduct, beta, result::Bra)
    tmp1 = Bra(b.operators[1].basis_r)
    operators.gemv!(Complex(1.)*b.factor, a, b.operators[1], Complex(0.), tmp1)
    for i=2:length(b.operators)-1
        tmp2 = Bra(b.operators[i].basis_r)
        operators.gemv!(Complex(1.), tmp1, b.operators[i], Complex(0.), tmp2)
        tmp1 = tmp2
    end
    operators.gemv!(alpha, tmp1, b.operators[end], beta, result)
end


end
