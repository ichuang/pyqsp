# Quantum computing with *gadgets* :pager: (a guide with examples)

Gadgets are unitary superoperators, based in the theory of quantum signal processing (QSP) and quantum singular value transformation (QSVT); they achieve flexible and intuitive block encodings of multivariable polynomial transforms. Uniquely, gadgets permit *function-first reasoning* about quantum programs, and offer improved resource complexity over competing methods for applying desired functions to the spectrum of large linear operators.

While running gadgets requires a quantum computer, their geometric form and realizing circuits can be described classically. This package allows the user to gadgets, snap them together into larger gadgets, visualize their output, and analyze their resource needs. 

Toward this, this README covers a variety of common methods for instantiating, modifying, combining, and visualizing gadgets. We pair these methods with coded examples wherever possible.

> :warning: This is a beta package; it is under active development. While we don't expect top-level methods and data structures to change, we are constantly working to improve performance, add quality-of-life features, and streamline the package. Major changes will be advertised where possible.

## Building gadgets :hammer:

Gadgets are unitary superoperators that take as input and return as output unitary matrices. But abstractly gadgets are just boxes with a finite number of input legs and output legs. Specifying them is simple.

```python
# Create two (2, 2) gadgets named "g0" and "g1".
g0 = Gadget(2, 2, "g0")
g1 = Gadget(2, 2, "g1")
```

Behind the scenes, the `Gadget` class will handle how these objects can and do connect, and later we will introduce a special sub-class of `Gadget`, `AtomicGadget` which can even be compiled to explicit circuits.

The `Gadget` class requires at least three arguments: `a`, `b`, and `label`. In the above example `Gadget(2, 2, "g0")` specifies `a = 2` input legs, `b = 2` output legs, and `label = "g0"`. We require `a, b >= 1` (though they can be different in general), and that labels are unique among gadgets. But don't worry: all of these properties are programmatically checked, and throw verbose errors if violated.

> :warning: *Gadgets cannot be named 'WIRE'*. For low-level reasons, this name is reserved, and if used will throw a verbose error.

## Linking gadgets together :link:

Gadgets can be linked together to form larger gadgets; to prevent self reference, `Gadget` objects are kept flat, and contain no internal gadget structure. Complex constructions keeping track of the internal structure of multiple gadgets are instead handled by the `GadgetAssemblage` class.

```python
# Wrap g0 and g1 into GadgetAssemblage objects, each containing a single gadget.
a0 = g0.wrap_gadget()
a1 = g1.wrap_gadget()

'''
Create a linking guide sending the 0-th output leg of g0 to the 0-th input of g1. 

Linking guides are lists of tuples; each element contains two tuples: (g0_name, output_leg_index) and (g1_name, input_leg_index).
'''
linking_guide = [(("g0", 0), ("g1", 0))]

# Create a GadgetAssemblage object containing two gadgets, linked according to linking_guide.
a2 = a0.link_assemblage(a1, linking guide)
```

Again, it is difficult to improperly specify assemblages of linked gadgets. Linking guides are checked and will throw a verbose error if improperly specified; later, we will expose easier ways to build linking guides in special cases.

While we have shown it already implicitly, multiple `GadgetAssemblage` objects, each containing multiple internal `Gadget` objects, can be simply combined. Internally, this is handled by the package unwrapping the internal `Gadget` objects contained in each `GadgetAssemblage`, updating their respective internal structure, and re-wrapping everything again with `GadgetAssemblage`. As such, a `GadgetAssemblage` object never contains reference to other `GadgetAssemblage` objects, only `Gadget` objects, `Interlink` (covered in the next section) objects, or their sub-classes.

### Tips and tricks when building gadgets :sun_with_face:

- The function `wrap_gadget` can be applied to both `Gadget` and `AtomicGadget` objects, and returns a `GadgetAssemblage` containing only that gadget, with all special internal attributes automatically defined. As `link_assemblage` expects `GadgetAssemblage` arguments, wrapping is often necessary when trying to append a single gadget.
- If one wishes to place a bunch of `Gadget` objects in parallel, then once can use `wrap_parallel_gadgets`; this function takes a simple list of `Gadget` objects, e.g., `[g0, g1, g2]`, and returns a `GadgetAssemblage` containing all gadgets in the list as if they had been linked with `[]` linking guides.
- A `GadgetAssemblage` has a `shape` attribute with the from `(a, b)`, where these `a` and `b` are the same as if the whole assemblage were being viewed as a `Gadget` object, forgetting internal structure.
- A `GadgetAssemblage` has a `input_dict` and `output_dict` attributes. As shown in the next section, these can be used to list off the `GadgetAssemblage` objects input and output legs for easy definition of a linking guide.
- A `GadgetAssemblage` can generate its own LaTeX `tikz` diagram! Given a `GadgetAssemblage` named `a0`, for instance, simply call `a0.print_assemblage()`. This returns a string which can be placed within a `tikzpicture` environment in LaTex, automatically creating a figure for the resulting gadget assemblage, showing gadget names, connecting wires, and internal structure. By default the plot uses colored wires, but calling it with the optional argument `in_color = False` turns this off.

## Gadget properties (for the curious) :wrench:

Behind the scenes, `GadgetAssemblage` objects are keeping track of a lot of structure. While for most purposes this structure is not and should not be directly touched, we provide a brief guide to relevant attributes and methods for the curious.

A `GadgetAssemblage` is primarily built from two simply ordered lists, one of of `Gadget` objects (named `gadgets`) and one of `Interlink` objects (named `interlinks`). When building a `GadgetAssemblage` object, these two lists are used, along with a series of `map_to_grid` dictionaries, to situate all `Gadget` and `Interlink` objects with respect to a dynamically generated 2D coordinate system, the `global_grid`. 

Specifying the right `map_to_grid` objects for each `Gadget` and `Interlink` object, such that a valid `GadgetAssemblage` is produced, is quite cumbersome, and we recommend you let this be handled programmatically by the `link_assemblage` method, and various wrapping functions. However, if you want to see a low-level definition of a `GadgetAssemblage` object, a few examples are given in `tests/test_gadget_assemblage.py`.

Remember that a `GadgetAssemblage`, if one forgot its internal structure, is just a gadget; aligned with this, the *effective* `a` and `b` of a `GadgetAssemblage` are easily recoverable.

```python
# Specify some GadgetAssemblage object.
assemblage = GadgetAssemblage(...)
# Retrieve the (automatically computed) effective shape
effective_a, effective_b = assemblage.shape
```

When linking complex `GadgetAssemblage` objects together, it can become difficult to remember their free input and output legs. Remember, a linking guide does not refer to `global_grid` (even though it modifies it behind the scenes); instead, it refers to free input and output legs of `GadgetAssemblage` objects. Listing these legs if forgotten is simple enough.

```python
# Specify some GadgetAssemblage object.
assemblage = GadgetAssemblage(...)
# Retrieve (automatically computed) input and output dicts.
input_leg_dict = assemblage.input_dict
output_leg_dict = assemblage.output_dict
'''
The keys of these dictionaries have the form (gadget_name, local_y_coord, local_x_coord).

Sometimes local_x_coord is helpful to know, but for linking guides, we don't need it, and can strip it off.
'''
input_legs = [e[:2] for e in list(input_leg_dict.keys())]
output_legs = [e[:2] for e in list(output_leg_dict.keys())]
```

As a reminder, input legs are labelled by the first `Gadget` *input* they encounter moving *forward*. Output legs are labelled by the first `Gadget` *output* they encounter moving *backward*.

## Atomic gadgets and generating circuits :electric_plug:

The `Gadget` class has a sub-class, `AtomicGadget`, which in addition to information about the number of input and output legs, takes information about how to run the corresponding unitary superoperator in terms of parallel instances of multivariable quantum signal processing (M-QSP) [\[see the paper here!\]](https://quantum-journal.org/papers/q-2022-09-20-811/).


```python
'''
Specify two length-2 lists of M-QSP phases (note anti-symmetry).

Here, xi_0 specifies the QSP phases for each of the two outputs of the eventual g0, in order. Here, the phases for each output are all zero.
'''
xi_0 = [[0, 0, 0, 0], [0, 0, 0, 0]]
xi_1 = [[0, 0, 0, 0], [0, 0, 0, 0]]

'''
Specify two length-2 lists of oracle guides (note symmetry).

Here, s_0 specifies the order in which each of the two input unitaries of the eventual g0 will be called. Here, the first output leg calls [0, 1, 0], while the second calls [1, 0, 1].
'''
s_0 = [[0, 1, 0], [1, 0, 1]]
s_1 = [[0, 1, 0], [1, 0, 1]] 

# Create two (2, 2) atomic gadgets named "g0" and "g1".
g0 = AtomicGadget(2, 2, "g0", xi_0, s_0)
g1 = AtomicGadget(2, 2, "g1", xi_0, s_1)

# Linking occurs as before, as AtomicGadget is a subclass of Gadget.
linking_guide = [(("g0", 0), ("g1", 0))]
a0 = g0.wrap_gadget()
a1 = g1.wrap_gadget()
a2 = a0.link_assemblage(a1, linking guide)
```

For a `GadgetAssemblage` object with shape `(a, b)`, each element *within* its `s` must contain only integers between `0` and `a - 1` inclusive. Moreover, the overall size of `s` must be of size `b`. For the same gadget, its `xi` must have the same length `b`, and each element *within* `xi` must have length *one greater* than the length of the corresponding element in `s`. Again, don't stress: these properties are checked programmatically, and will throw verbose errors.
 
One more thing: in order for the functional transforms achieved by atomic gadgets to be real (and thus semantically useful), `xi` and `s` for each atomic gadget should satisfy basic symmetries. That is, each element of `s` should be symmetric (the same back-to-front), while each element of `xi` should be antisymmetric (it goes to its negation under reversal).

In special cases, these symmetry constraints can be broken, leading to *atypical atomic gadgets*, but we leave discussion of this to a later section on [additional methods](#additional-attributes-and-methods).

If a `GadgetAssemblage` comprises only `AtomicGadget` objects, additional functionalities are enabled, chiefly the ability to realize the resulting composite gadget as a circuit with a QSP-like form. Such a circuit, if it can exist, is automatically generated on instantiation of both `AtomicGadget` and `GadgetAssemblage` objects, and can be referenced through their `sequence` attributes.

```python
# An atomic gadget has its own sequence.
g0 = AtomicGadget(1, 1, "g0", [[0, 0, 0, 0]], [[0, 0, 0]])
gadget_sequence = g0.sequence

# The wrapped version of a0 contains the same sequence in this case, but in general will depend on the arrangement of internal AtomicGadget objects.
a0 = g0.wrap_gadget()
assemblage_sequence = a0.sequence
```

In both cases, `sequence` is a length `b` *list of lists of `SequenceObject` objects*. Each of these lists can be read off, left to right, to generate the corresponding quantum circuit, time going again left to right, for that output. Each `SequenceObject` represents a single-qubit or two-qubit gate, possibly controlled on other qubits, and possibly representing an unknown oracle unitary. Rarely, however, does one need to work with this circuit representation explicitly from the `SequenceObject` lists. Instead, these objects are almost always only used by internal methods of the package to yield Qiskit-based quantum circuit objects.

### On compiling gadget assemblages to Qiskit :computer:

The main logic used to turn `sequence` into a circuit is contained in `seq2circ.py`; the main method within this module, `seq2circ`, can take either a list of lists of `SequenceObject` objects, or just a list of `SequenceObject` objects, and product a quantum circuit using Qiskit.

```python
# Create a gadget g0.
g0 = AtomicGadget(1, 1, "g0", [[0, 0, 0, 0]], [[0, 0, 0]])
# Wrap gadget g0 to an assemblage (optional here).
a0 = g0.wrap_gadget()
# Get sequence for assemblage a0.
assemblage_sequence = a0.sequence
# Get quantum circuit from assemblage_sequence.
q_circ = seq2circ(assemblage_sequence, verbose=False)
```

A core visualization technique in QSP is the *response function*; that is, as input oracles in QSP/M-QSP are single-qubit rotations parameterized by a scalar value `x`, it is useful and easy to plot matrix elements of the resulting protocol with respect to changing `x`. A variety of sophisticated visualization techniques for gadgets are possible, and we give the simplest below.

```python
# Standard python math and plotting imports.
import numpy as np
from matplotlib import pyplot as plt
# Generate quantum circuit, and produce discrete data.
q_circ = seq2circ(assemblage_sequence, verbose=False)
# X is real within [-1, 1] and Y has modulus within [-1, 1].
X, Y = q_circ.one_dim_response_function(npts=50)
```

The simplest function `one_dim_response_function` evaluates the gadget on a scalar parameter `x` *for all `a` inputs at once*. By default, `Y`, which has the same size as `X`, is a list of `00` matrix elements of the resulting unitary. In the section on [additional methods](#additional-attributes-and-methods), we indicate how one can choose to query an arbitrary matrix element, as well as sweep gadget outputs over more complex input sets.

### Tips and tricks when building atomic gadgets :sun_with_face:

- `AtomicGadget` objects can be cast to a string form. This will list the indices of output legs of the `AtomicGadget`, followed by a human-readable list of the `SequenceObject` objects constituting that leg. Note that for high-degree gadgets, these strings might be quite long. `GadgetAssemblage` objects comprising only `AtomicGadget` objects can also be cast to strings, but may produce prohibitively long strings. The string form of a generic `GadgetAssemblage` object just lists the names of all contained gadgets, joined by hyphens, e.g., `g0-g1-g2`.
- For faster plotting, the `npts` argument in `one_dim_response_function` can be reduced; this number is precisely the number of equal divisions of `[-1, 1]` over which input `x` are plotted.

## Additional attributes and methods

In what follows we describe a variety of more involved optional arguments, suppressed properties, and corresponding reminders for the advanced user of this package. Many of these points will eventually be defined, handled, or checked automatically. We will indicate explicitly where and when this occurs.

- While all `AtomicGadget` objects with antisymmetric `Xi` and symmetric `S` are guaranteed to have real `00` matrix elements for each output leg, this is not true in general. However, there exist  `AtomicGadget` objects without antisymmetric `Xi` and symmetric `S` which *do* have this property. Currently no checks are in place to verify whether this is the case, so users are warned to verify this property (or an approximate version of it) independently is using such *atypical atomic gadgets*.
- The function Unitary `one_dim_response_function` can take an optional argument, `uindex` of the form `(int, int)`, indicating which matrix element should be returned from the resulting matrix, evaluated over the size-`npts` set `X`. By default `uindex = (0, 0)`.
- The `AtomicGadget` class currently has two optional arguments corresponding to internal attributes which are currently undefined. The first `correction_guide`, will allow for different *output legs* of an atomic gadget to receive different-length correction protocols when `sequence` is being generated. Currently all outgoing legs are corrected with the same protocol, fixed at `degree=4` in `get_correction_phases` in `gadget_assemblage.py`. The second attribute, `pinning_guide`, will allow for *input legs* of an atomic gadget to have pre-fixed input values. In other words, the free values for the user to defined for the input legs of a `GadgetAssemblage` containing such *pinned* `AtomicGadget` objects will be fewer in number. The user is already free to fix a subset of a `GadgetAssemblage` object's inputs, but eventually this will be handled and checked at the individual gadget level, helping in the definition of certain common *atypical atomic gadgets*.

## Coming soon :construction:
- Improved ancillae-allocation subroutines in `gadget_assemblage.py`, taking advantage of special properties of the `AtomicGadget` object `S` attribute.
- Instantiation and automatic verification of `correction_guide` attribute in `AtomicGadget`.
- Instantiation and automatic verification of `pinning_guide` attribute in `AtomicGadget`.