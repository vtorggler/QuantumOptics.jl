.. _section-bases:

Bases
=====

The primary purpose of bases in **Quantumoptics.jl** is to specify the dimension of the Hilbert space of the system and to make sure that quantum objects associated to distinct bases can't be combined accidentally in an incorrect way. Many of the commonly used basis types like

* :ref:`section-spin`
* :ref:`section-fock`
* :ref:`section-particle`

are already built into **Quantumoptics.jl**. Hilbert spaces of composite systems can be handled automatically with the :jl:type:`CompositeBasis` which can for example be created using the `tensor` function::

    basis_fock = FockBasis(10)
    basis_particle = MomentumBasis(0., 10., 50)
    basis = tensor(basis_fock, basis_particle)

For cases not covered by these one can either use the :jl:type:`bases.GenericBasis` or implement own special purpose bases by deriving from the abstract :jl:abstract:`Basis` type. The only mandatory property of all basis types is that they have a field `shape` which specifies the dimensionality of their Hilbert space. E.g. a spin 1/2 basis could be implemented as::

    type SpinBasis <: Basis
        shape::Vector{Int}
        SpinBasis() = new(Int[2]) # Constructor
    end

The interaction with other bases can be determined by overloading the `==` operator as well as the :jl:func:`bases.multiplicable` function::

    ==(b1::SpinBasis, b2::SpinBasis) = true
    multiplicable(b1::SpinBasis, b2::SpinBasis) = true


.. toctree::
    :hidden:

    spin
    fock
    particle