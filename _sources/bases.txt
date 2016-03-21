.. _section-bases:

Bases
=====

The primary purpose of bases in **Quantumoptics.jl** is to specify the dimension of the Hilbert space of the system and to make sure that quantum objects associated to distinct bases can't be combined accidentally in an incorrect way. Many of the common types of bases used in quantum mechanics like

* :ref:`Spin basis <section-spin>`
* :ref:`Fock basis <section-fock>`
* :ref:`Position basis and Momentum basis <section-particle>`

are already imeplmented. The are treated in more detail in the section :ref:`section-quantumsystems`.


Composite bases
---------------

Hilbert spaces of composite systems can be handled automatically with the :jl:type:`CompositeBasis` which can for example be created using the `tensor` function::

    basis_fock = FockBasis(10)
    basis_particle = MomentumBasis(0., 10., 50)
    basis = tensor(basis_fock, basis_particle)


Subspace basis
--------------

Restricting a Hilbert space to a subspace is done using a :jl:type:`SubspaceBasis`. It is defined by N, not necessarily orthogonal states that live in the embedding Hilbert space.

.. math::

    | x^\prime \rangle
            = \sum_{u \in subspace}
                    |u_o \rangle \langle u^o
                        | x \rangle

Any SubspaceBasis can be made orthogonal with help of the :jl:func:`orthonormalize` function which utilizes the numerical stable modified Gram-Schmidt algorithm.

Changing between the superbasis and any subspaces bases can be done by first creating a projection operator with the :jl:func:`projector(::SubspaceBasis, ::Basis)` function::

    b = FockBasis(5)
    b_sub = SubspaceBasis(b, [fockstate(b, 1), fockstate(b, 2)])

    P = projector(b_sub, b)

    x = coherentstate(b, 0.5)
    x_sub1 = P*x

The projection operation is irreversible if the original state was not already contained in the subspace. However, it is of course possible to represent any state in the subspace in the superbasis::

    y = dagger(P)*x_sub1 # Not equal to x


Generic bases
-------------

If the needed basis type is not available the quick and dirty way is to use a :jl:type:`GenericBasis`. It just needs to know the dimension of the Hilbert space and its ready to go::

    b = GenericBasis(5)

However, since operators and states represented in any generic basis can be combined as long as the bases have the same dimension it might lead to errors that otherwise could have been caught easily.


Implementing new bases
----------------------

The cleaner way is to implement own special purpose bases by deriving from the abstract :jl:abstract:`Basis` type. The only mandatory property of all basis types is that they have a field `shape` which specifies the dimensionality of their Hilbert space. E.g. a spin 1/2 basis could be implemented as::

    type SpinBasis <: Basis
        shape::Vector{Int}
        SpinBasis() = new(Int[2]) # Constructor
    end

The interaction with other bases can be determined by overloading the `==` operator as well as the :jl:func:`bases.multiplicable` function which allow to control the behaviour when interaction with other bases.
