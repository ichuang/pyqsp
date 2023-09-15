#######################################
#######################################
#######################################
"""
Notes on implementation:

This class will serve to replace the CompositeGadget class entirely; beyond changing get_sequence(), the GadgetAssemblage class can use the other methods defined previously in CompositeGadget.

Right now the only ambiguity in the get_sequence() method is in determining input_y, which should be the global_grid coordinate of the input leg of the last gadget corresponding to the oracle indicated by external_seq[j]. If there is an easy way to get an integer 'n' out of external_seq[j] between 0 and a-1 for the last gadget, then input_y should be last_gadget.map_to_grid[(n, 0)][0].

The other missing component is in the final branch with the comment "# It's possible from our definitions that this branch is never activated, but I am not sure." This should be the case that a wire at a given location misses the last gadget and goes all the way to the beginning of the assemblage (to the left) without hitting another gadget. In this case, we should just append a bare oracle call according to a similar procedure as above.

Otherwise, the Gadget and AtomicGadget objects are compatible, up to adding Jack's 'input_legs' and 'output_legs' to their init classes. These shouldn't interfere with one another.

I have added an extra 'depth' parameter to get_sequence(), which is increased whenever it is recursively called; note this may be different from Jack's intended value.

I have also not yet added cosmetic global checks indicated in the statement of get_sequence(), like that one is actually querying a point on in global_grid, etc.

For examples on instantiating gadgets, see the simplest example of non-linked gadgets, like below:

#######################################

    g0 = Gadget(1, 2, "g0")
    g1 = Gadget(1, 2, "g1")
    g2 = Gadget(1, 2, "g2")

    a0 = g0.wrap_gadget() # make bare assemblage objects
    a1 = g1.wrap_gadget() # make bare assemblage objects
    a2 = g2.wrap_gadget() # make bare assemblage objects

    a3 = a0.link_assemblage(a1, []) # use trivial interlinks
    a4 = a3.link_assemblage(a2, []) # use trivial interlinks

    print()
    print("JOINED GADGETS EXPRESSIBLE FORMAT")
    print(a4.print_assemblage())

#######################################

If you want a non-trivial interlink, you can make replace the empty list in the first link_assemblage, for instance, with [(("g0", 0), ("g1", 0))], which will link the 0th output leg of g0 with the 0th input leg of g1. Running this with the current code will produce tikz output (after a lot of debugging print statements).

"""
#######################################
#######################################
#######################################

class GadgetAssemblage(): # This does not need to inherit from gadget, and in fact I think it shouldn't, because we shouldn't be able to stick them inside other GadgetAssemblage objects.
    
    """
    The rest of the class is not shown in this template.
    """

    def get_sequence(self, global_grid_y, global_grid_x, depth, correction=None):

        # First we check that all gadgets are atomic gadgets (NOT IMPLEMENTED).
        # Then we check that we're in the global grid keyset (NOT IMPLEMENTED).
        # Then we check that global grid loc doesn't map to something trivial (NOT IMPLEMENTED).
        # The above checks are not crucial, but will help users debug.
        # Then we continue with the structure of the original get_sequence method from Jack's code.

        last_gadget_output = set()

        # We create a set of output legs for the final gadget.
        for y in range(len(self.global_grid)):
            if self.gadget_dict[(y, global_grid_x)][0][0] != "WIRE":
                last_gadget_output.add((y, global_grid_x))
            else:
                continue

        # We check if we are at the last gadget's output legs or not.
        if (global_grid_y, global_grid_x) in last_gadget_output:
            # Retrieve last gadget object by its name.
            last_gadget = self.gadget_set[self.gadget_dict[(global_grid_y, global_grid_x)][0][0]]
            # Get the sequence for the last gadget (assuming atomic for now).
            external_seq = last_gadget.get_sequence()
            # Instantiate an empty sequence.
            seq = []

            # Loop over sequence.
            for j in range(len(external_seq)):
                # If it is an oracle, we find out which one.
                if isinstance(external_seq[j], QSP_Signal):
                    
                    # We need to find which signal leg this is
                    input_leg_index = external_seq[j].label
                    
                    # UNFIXED: Input y should be the global y coordinate of the input leg of the last gadget corresponding to the oracle at external_seq[j].
                    input_y = None # CURRENTLY NOT INSTANTIATED
                    # Input x is just the global x coordinate of the last gadget.
                    input_x = last_gadget.map_to_grid[(0, 0)][1]

                    # Find out where that particular input leg of the last gadget maps backwards to
                    next_collision = self.origin_guide[(input_y, input_x)]

                    # Check if particular input leg maps backwards to gadget.
                    if next_collision[0] != "WIRE":
                        # Get the gadget the wire collides with
                        collision_gadget = self.gadget_set[next_collision[0]]
                        # Get the global grid position of the collision location
                        c_y, c_x = collision_gadget.map_to_grid[(next_collision[1], 1)]

                        # UNFIXED: this should determine if a given output leg needs to be corrected or not, but the index here (global grid based) may not be the correct one.
                        corr = self.correction[c_y]

                        # Make the recursive call at this position with increased depth.
                        internal_seq = self.get_sequence(c_y, c_x, depth + 1, correction=corr)
                        # Finally, tack on the internal sequence
                        seq.extend(internal_seq)
                    # If input legs map back to overall input leg instead.
                    else:
                        # NOTE: In this case it seems like we want to just insert a standard oracle; but is this the right one? We probably want to insert one as indexed by the global grid.
                        seq.append(external_seq[j])
                else:
                    # If not a signal, just insert the relevant phase.
                    seq.append(external_seq[j])

        # If the queried position does not encounter last gadget, but something further back.
        else:
            # Find the thing it runs into.
            next_collision = self.origin_guide[(global_grid_y, global_grid_x)]
            # If that thing is not an overall input wire.
            if next_collision[0] != "WIRE":
                # Get the gadget the wire collides with
                collision_gadget = self.gadget_set[next_collision[0]]
                # Get the global grid position of the collision location
                c_y, c_x = collision_gadget.map_to_grid[(next_collision[1], 1)]
                # Call the function with new position
                seq = self.get_sequence(c_y, c_x, depth + 1, correction=corr)
            # If the thing is an overall input wire.
            else:
                # UNFIXED: This means the position doesn't encounter the last gadget, and instead leads to an input wire for the entire assemblage. In other words, we should just append a bare oracle call here at the right location.

                # It's possible from our definitions that this branch is never activated, but I am not sure.
                pass

        # Finally, wrap the output sequence if correction indicated.
        if correction:
            seq = corrected_sequence(seq, depth, correction)
            return seq
        else:
            return seq

#######################################
#######################################
#######################################