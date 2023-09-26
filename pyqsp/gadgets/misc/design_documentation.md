# Overview of design choices

## File structure
- [x] `gadget_assemblage.py` contains the `GadgetAssemblage` class by itself, and imports the more basic objects of `Gadget` and `AtomicGadget`.
- [ ] `gadget_components.py` contains the basic objects of `Gadget` and `AtomicGadget`.
- [x] Tests are located in the `tests` directory. Tests can be run with `python -m unittest test.test_gadget_assemblage`.

## Enumerated classes
- The `Gadget` class is a superclass of `AtomicGadget` only; the former contains only size information and derived properties, while the latter contains phase and oracle information, as well as a variety of methods for viewing internal structure, and so on.
- `AtomicGadget` as mentioned contains phase and oracle information. We discuss quality of life methods below.
	- Each `AtomicGadget` should be able to output its defining sequence; this is an uncorrected, standard M-QSP protocol, and contains only Z and X rotations, properly labelled. There is no mention of ancillae, as these objects are flat.
	- Each element in the sequence can have its own class; these are Z and X rotations, SWAPS, and oracle unitaries. We also want these classes to contain information about control targets and sources (the latter can have many indices). For `AtomicGadget` objects this control information is trivial and necessarily ignored; such information is only instantiated when these objects are inside a `GadgetAssemblage` object, and its `get_sequence` has been called.
- The `Interlink` class looks superficially similar to the `Gadget` class, but only because it specifies a permutation on the global grid to which `Gadget` objects are also adhered.
- The `GadgetAssemblage` class contains an ordered list of `Gadget` and `Interlink` objects, and checks whether they specify a valid causal network, agnostic to whether any of the underlying `Gadget` object are atomic or not. If all underlying gadgets are atomic, then one can call a series of additional methods related to the circuit representation of the `GadgetAssemblage`. This class is flat in the sense that it does not inherit from `Gadget`, even though a `GadgetAssemblage` can technically be viewed as a gadget (and has methods for computing size, input and output legs, etc.).
	- The important method for `GadgetAssemblage` are `link_assemblage()`, which allows one to link two assemblages, unwrapping their contents, joining them, and re-wrapping them in a `GadgetAssemblage` object. Doing this automatically rechecks the validity of the resulting `GadgetAssemblage`.
	- Various quality of life methods exist for building assemblages; parallel non-linked gadgets can be wrapped into an assemblage all at once using `wrap_parallel_gadgets()`, while a single gadget can be wrapped through a method in the `Gadget` class, `wrap_gadget()`. Examples of all of these actions are found in the tests.

## Higher level notes on correction
- Correction is only seen in `GadgetAssemblage`; these can contain one gadget, and so this subsumes the almost trivial case. As such, correction should not be a fundamental property of a gadget, nor a specific gadget, but an effect on the circuit output when compiling a `GadgetAssemblage` containing only `AtomicGadget` objects to a circuit/circuit description.
- For ancillae; one can compute the maximum width of a circuit before generating its description by calling a method on a `GadgetAssemblage` object. This information can be used to initialize an ancilla register fixed beforehand. At a given depth during traversal of a `GadgetAssemblage` object during its `get_sequence()` method, one of these ancillae will be privileged and used in the controllization of an approximate Z rotation.
- To keep ancillae allocation simple, we keep all gates applied to the zero-index qubit (the one that would be used for a standard QSP protocol), and all (possibly multi) controls at higher-index ancilla. As we are often interested in phase-kickbacks, however (see Fig. 5 in the main paper), we will have to conjugate these operations with SWAPS.
- For the moment it may be easier to use the same depth correction protocol for every gadget output; we could however include a flag in each `AtomicGadget` object which indicates how corrected it should be (as a polynomial degree); this would be a quick change, but isn't needed for initial efforts.

## On handling multiple controls
- Correction of a gadget involves extraction first, i.e., using a gadget to generate an approximate Z rotation; this Z rotation is then controllized and combined with the original gadget sequence to produce a corrected output
- `Can the controllization process at a given depth be handled by simply adding all controls to a given sequence element up to that specified depth?` Answer: yes, and this has been implemented.
- The resulting circuit will look, at each step, to an outside observer, like a Z conjugated X rotation; when correcting, we run extraction and then controllize the whole thing; for our purposes, this means adding a control on top of any controls already in existence. The SWAPS at the far ends will take care of moving the accrued half phase to the right qubit.
- `Does this mean we have to pass arguments inside get_sequence() for ancillae tracking, or can we derive everything from the depth? It seems like the depth during traversal tells us everything we need to know.` Everything can be derived from depth, and is handled in the course of the recursive call automatically.

## Interaction with quantum circuit packages
- The first step is to create an unambiguous description of the quantum circuit for a gadget; compiling this to an actual unitary will be done separately (and hopefully simply).
- Handling multi controls in Qiskit seems to be relatively simple, for instance see [here](https://quantumcomputing.stackexchange.com/questions/11932/how-to-make-circuit-for-n-control-z-gate-i-e-c3z), or consult the official documentation.

## Remaining tasks
- [x] Function on `GadgetAssemblage` object returning maximum integer depth.
- [x] Writing up `get_gadget_sequence` for `AtomicGadget`
- [x] Writing up `get_assemblage_sequence` for `GadgetAssemblage`
- [x] Writing a check-function for atomic assemblage definition; right now any signals are allowed, which can throw errors if one doesn't define the oracles properly.
- [x] Currently the swap functions (unused) do not preserve atomic gadgets; this can be done using the same method that `link_assemblage` uses. 
- [x] `wrap_gadget` can be modified to handle both standard an atomic gadgets, rather than using two methods with `wrap_atomic_gadget`.
- [x] Adding a new argument for `get_correction_phases` to specify polynomial degree; eventually this will be a property assigned to each atomic gadget for its output legs individually, but for now a uniform value is fine.
- [x] Implementing square root gadget, again relying on the phases computed by Jack's code within the new framework. This may require a preceeding and following shift. A flag can be assigned to an atomic gadget which applies a shift on all of its inner oracles (simple, as these are treated differently from signals) as well as the reverse shift on its `corrected output`, which is encountered during traversal backwards through an assemblage.
- [ ] It's possible for one to desire that a given wire be 'shifted'; that is, the approximate X rotation has its value modified by a known amount. This can be a property of an input or output leg (post-correction) of a gadget, or it can be treated as an uncorrected (1,1) gadget and tacked on in various places. For the moment we will ignore it, but it is an easy thing to include.
- [ ] Right now the code is being overly conservative with ancilla allocation; depending on `S` for a given `AtomicGadget`, certain paths backwards through the gadget may not be taken at all, limiting the maximum depth seen by that leg; moreover, if `S` is a single variable sequence, correction can be deferred to the following gadget in general. Each of these can be simply accomodated by changing how `max_depth` runs.
- [ ] Individual gadgets can have both `pinning_guide` and `correction_guide` attributes specifying the degree to which a given output leg needs to be corrected, as well as whether any input legs are to be automatically fixed when casting to a untiary. These are currently not used within the main body of the code.
- [ ] Readme and further docstrings can be added where appropriate.
- [ ] Moving the gadget and atomic gadget definitions to their own file, perhaps along with the top-level methods. 
- [ ] Multidimensional plotting functionality. We always target a specific matrix element for now, but we should be able to produce either multidimensional arrays or a huge flat list of tuples which plots over relevant slices. What is the best way to do this? Vector basis in each coordinate, following the syntax currently in seq2circ.py?

## Bugs to fix
- [ ] Currently correction and pinning guides need to be replaced upon gadget wrapping (fixed already) and swapping.
- [ ] For (1,1) gadgets that feed into other (1,1) gadgets, correction can always be neglected, and this could be dynamically checked.
- [ ] Setting up checks on correction guide to ensure uniqueness and range specifications. These checks are: unique keys, keys within output leg range, values positive integers greater than or equal to four.
- [ ] Plotting options should be expanded; sweep and fixed, for arbitrary slices. If anything, just allow one to feed in a series of a-dimensional input point, and return these points evaluated through the gadget as d-dimensional outputs.
- [ ] Pinning guides requires more work to get up and running, and should interact nicely with current plotting.
- [ ] Changing ancilla allocation to the new, more conservative form, which takes in to account atomic gadget S prescriptions.
