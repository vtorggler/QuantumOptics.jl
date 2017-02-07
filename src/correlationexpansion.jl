module correlationexpansion

import Base: trace, ==, +, -, *, /
import ..operators: dagger, identityoperator,
                    trace, ptrace, normalize!, tensor, permutesystems,
                    gemv!, gemm!

using Combinatorics, Iterators
using ..bases
# using ..states
using ..operators
using ..operators_dense
using ..operators_lazy
# using ..ode_dopri

# import Base: *, full
# import ..operators

typealias Mask{N} NTuple{N, Bool}

indices2mask(N::Int, indices::Vector{Int}) = Mask(tuple([(i in indices) for i=1:N]...))
mask2indices{N}(mask::Mask{N}) = Int[i for i=1:N if mask[i]]

complement(N::Int, indices::Vector{Int}) = Int[i for i=1:N if i ∉ indices]
complement{N}(mask::Mask{N}) = tuple([! x for x in mask]...)

correlationindices(N::Int, order::Int) = Set(combinations(1:N, order))
correlationmasks(N::Int, order::Int) = Set(indices2mask(N, indices) for indices in
        correlationindices(N, order))
correlationmasks{N}(S::Set{Mask{N}}, order::Int) = Set(s for s in S if sum(s)==order)
subcorrelationmasks{N}(mask::Mask{N}) = Set(indices2mask(N, indices) for indices in
        chain([combinations(mask2indices(mask), k) for k=2:sum(mask)-1]...))


"""
An operator including only certain correlations.

operators
    A tuple containing the reduced density matrix of each subsystem.
correlations
    A (mask->operator) dict. A mask is a tuple containing booleans which
    indicate if the corresponding subsystem is included in the correlation.
    The operator is the correlation between the specified subsystems.
"""
type ApproximateOperator{N} <: Operator
    basis_l::CompositeBasis
    basis_r::CompositeBasis
    factor::Complex128
    operators::NTuple{N, Operator}
    correlations::Dict{Mask{N}, Operator}

    function ApproximateOperator{N}(basis_l::CompositeBasis, basis_r::CompositeBasis,
                operators::NTuple{N, DenseOperator},
                correlations::Dict{Mask{N}, DenseOperator},
                factor::Number=1.)
        @assert N == length(basis_l.bases) == length(basis_r.bases)
        for i=1:N
            @assert operators[i].basis_l == basis_l.bases[i]
            @assert operators[i].basis_r == basis_r.bases[i]
        end
        for (mask, op) in correlations
            @assert sum(mask) > 1
            @assert op.basis_l == tensor(basis_l.bases[[mask...]]...)
            @assert op.basis_r == tensor(basis_r.bases[[mask...]]...)
        end
        new(basis_l, basis_r, factor, operators, correlations)
    end
end

function ApproximateOperator{N}(basis_l::CompositeBasis, basis_r::CompositeBasis, S::Set{Mask{N}})
    operators = ([DenseOperator(basis_l.bases[i], basis_r.bases[i]) for i=1:N]...)
    correlations = Dict{Mask{N}, DenseOperator}()
    for mask in S
        @assert sum(mask) > 1
        correlations[mask] = tensor(operators[[mask...]]...)
    end
    ApproximateOperator{N}(basis_l, basis_r, operators, correlations)
end

ApproximateOperator{N}(basis::CompositeBasis, S::Set{Mask{N}}) = ApproximateOperator(basis, basis, S)
function ApproximateOperator{N}(operators::Vector, S::Set{Mask{N}})
    @assert length(operators) == N
    correlations = Dict{Mask{N}, DenseOperator}()
    for op in operators
        @assert typeof(op) <: Operator
    end
    for mask in S
        @assert sum(mask) > 1
        b_l = CompositeBasis([op.basis_l for op in operators[[mask...]]]...)
        b_r = CompositeBasis([op.basis_r for op in operators[[mask...]]]...)
        correlations[mask] = DenseOperator(b_l, b_r)
    end
    b_l = CompositeBasis([op.basis_l for op in operators]...)
    b_r = CompositeBasis([op.basis_r for op in operators]...)
    ApproximateOperator{N}(b_l, b_r, (operators...), correlations)
end

"""
Tensor product of a correlation and the density operators of the other subsystems.

Arguments
---------
operators
    Tuple containing the reduced density operators of each subsystem.
mask
    A tuple containing booleans specifying if the n-th subsystem is included
    in the correlation.
correlation
    Correlation operator for the subsystems specified by the given mask.
"""
function embedcorrelation{N}(operators::NTuple{N, DenseOperator}, mask::Mask{N},
            correlation::DenseOperator)
    # Product density operator of all subsystems not included in the correlation.
    if sum(mask) == N
        return correlation
    end
    ρ = tensor(operators[[complement(mask)...]]...)
    op = correlation ⊗ ρ # Subsystems are now in wrong order
    perm = sortperm([mask2indices(mask); mask2indices(complement(mask))])
    permutesystems(op, perm)
end
embedcorrelation{N}(operators::NTuple{N, DenseOperator}, indices::Vector{Int}, correlation::DenseOperator) = embedcorrelation(operators, indices2mask(indices), correlation)

"""
Calculate the normalized correlation of the subsystems specified by the given index mask.

Arguments
---------
rho
    Density operator of the total system.
mask
    A tuple containing booleans specifying if the n-th subsystem is included
    in the correlation.

Optional Arguments
------------------
operators
    A tuple containing the reduced density operators of the single subsystems.
subcorrelations
    A (mask->operator) dictionary storing already calculated correlations.
"""
function correlation{N}(rho::DenseOperator, mask::Mask{N};
            operators::NTuple{N, DenseOperator}=([ptrace(normalize(rho), complement(N, [i]))
                                                  for i in 1:N]...),
            subcorrelations::Dict{Mask{N}, DenseOperator}=Dict())
    # Check if this correlation was already calculated.
    if mask in keys(subcorrelations)
        return subcorrelations[mask]
    end
    order = sum(mask)
    rho = normalize(rho)
    σ = ptrace(rho, mask2indices(complement(mask)))
    σ -= tensor(operators[[mask...]]...)
    for submask in subcorrelationmasks(mask)
        subcorrelation = correlation(rho, submask;
                                     operators=operators,
                                     subcorrelations=subcorrelations)
        σ -= embedcorrelation((operators[[mask...]]...), submask[[mask...]], subcorrelation)
    end
    subcorrelations[mask] = σ
    σ
end

correlation{N}(rho::ApproximateOperator{N}, mask::Mask{N}) = rho.correlations[mask]


"""
Approximate a density operator by including only certain correlations.

Arguments
---------
rho
    The density operator that should be approximated.
masks
    A set containing an index mask for every correlation that should be
    included. A index mask is a tuple consisting of booleans which indicate
    if the n-th subsystem is included in the correlation.
"""
function approximate{N}(rho::DenseOperator, masks::Set{Mask{N}})
    alpha = trace(rho)
    rho = normalize(rho)
    operators = ([ptrace(rho, complement(N, [i])) for i in 1:N]...)
    subcorrelations = Dict{Mask{N}, DenseOperator}() # Dictionary to store intermediate results
    correlations = Dict{Mask{N}, DenseOperator}()
    for mask in masks
        correlations[mask] = correlation(rho, mask;
                                         operators=operators,
                                         subcorrelations=subcorrelations)
    end
    ApproximateOperator{N}(rho.basis_l, rho.basis_r, operators, correlations, alpha)
end

ptrace{N}(mask::Mask{N}, indices::Vector{Int}) = (mask[complement(indices)]...)

function ptrace{N}(rho::ApproximateOperator{N}, indices::Vector{Int})
    operators = (rho.operators[complement(indices)]...)
    factors = [trace(op) for op in rho.operators]
    result = tensor(operators...)
    for mask in keys(rho.correlations)
        I = mask2indices(mask)
        J = I ∩ indices
        if isempty(J)
            correlation = rho.correlations[mask]
        else
            correlation = ptrace(rho.correlations[mask], J)
        end
        factor = prod(factors[setdiff(indices, I)])
        result += factor*embedcorrelation(operators, ptrace(mask, indices), correlation)
    end
    rho.factor*result
end

function full{N}(rho::ApproximateOperator{N})
    result = tensor(rho.operators...)
    for (mask, correlation) in rho.correlations
        result += embedcorrelation(rho.operators, mask, correlation)
    end
    rho.factor*result
end

function *{N}(rho::ApproximateOperator{N}, op::LazyTensor)
    operators = ([i ∈ op.operators ? rho.operators[i]*op.operators[i] : copy(rho.operators[i]) for i=1:N]...)
    correlations = Dict{Mask{N}, DenseOperator}()
    for mask in keys(rho.correlations)
        I = mask2indices(mask)
        D = Dict(i=>op.operators[i] for i in I ∩ keys(op.operators))
        if isempty(D)
            correlations[mask] = rho.correlations[mask]
        else
            I_ = operators.complement(I)
            op_I = embed(ptrace(op.basis_l, I_), ptrace(op.basis_r, I_), D)
            correlations[mask] = rho.correlations[mask]*op_I
        end
    end
    ApproximateOperator{N}(basis_l, basis_r, operators, correlations)
end

function *{N}(op::LazyTensor, rho::ApproximateOperator{N})
    operator_list = ([i ∈ keys(op.operators) ? rho.operators[i]*op.operators[i] : copy(rho.operators[i]) for i=1:N]...)
    correlations = Dict{Mask{N}, DenseOperator}()
    for mask in keys(rho.correlations)
        I = mask2indices(mask)
        D = Dict(i=>op.operators[i] for i in I ∩ keys(op.operators))
        if isempty(D)
            correlations[mask] = rho.correlations[mask]
        else
            I_ = operators.complement(N, I)
            op_I = embed(ptrace(op.basis_l, I_), ptrace(op.basis_r, I_), operators_lazy.removeindices(D, I_))
            correlations[mask] = op_I*rho.correlations[mask]
        end
    end
    ApproximateOperator{N}(basis_l, basis_r, operator_list, correlations)
end

function dmaster{N}(rho::ApproximateOperator{N}, H::LazySum,
                    Gamma::Matrix{Complex128}, J::Vector{LazyTensor}, Jdagger::Vector{LazyTensor})
    A = LazySum([a*(h*rho) for (a, h) in zip(H.factors, H.operators)])
    A -= LazySum([a*(rho*h) for (a, h) in zip(H.factors, H.operators)])
    for j=1:length(J), i=1:length(J)
        A += Gamma[i,j]*(J[i]*rho*Jdagger[j])
        A -= Gamma[i,j]*0.5*(Jdagger[j]*(J[i]*rho))
        A -= Gamma[i,j]*0.5*((rho*Jdagger[j])*J[i])
    end
    doperators = ([ptrace(A, complement(N, [i])) for i=1:N]...)
    dcorrelations = Dict{Mask{N}, DenseOperator}()
    for order=2:N
        for mask in correlationmasks(Set(keys(A.correlations)), order)
            I = mask2indices(mask)
            suboperators = rho.operators[I]
            # Tr{̇̇d/dt ρ}
            σ_I = ptrace(A, complement(I))
            # d/dt ρ^{s_k}
            for i = 1:order
                σ_I -= embedcorrelation(suboperators, [i], doperators[I[i]])
            end
            # d/dt σ^{s}
            for submask in keys(dcorrelations) ∩ subcorrelationmasks(mask)
                σ_I -= embedcorrelation(suboperators, ptrace(submask, I), dcorrelations[submask])
                for i in setdiff(complement(N, I), mask2indices(complement(mask)))
                    ops = ([i==j ? doperators[j] : rho.operators[j] for j in I]...)
                    σ_I -= embedcorrelation(ops, ptrace(submask, I), operators.correlations[submask])
                end
            end
        end
    end
    return ApproximateOperator(rho.basis_l, rho.basis_r, doperators, dcorrelations, rho.factor)
end

end # module