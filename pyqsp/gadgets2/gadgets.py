import copy

class Gadget:

    def __init__(self, a, b, label, map_to_grid=dict()):
        self.a, self.b = a, b
        self.label = label
        self.legs = [(k, 0) for k in range(a)] + [(k, 1) for k in range(b)]
        self.map_to_grid = map_to_grid

        # THIS IS AN ADDITION FOR COMPATIBILITY W/ JACK'S GADGET CLASS
        self.in_labels, self.out_labels = [(label, j) for j in range(a)], [(label, j) for j in range(b)]

    # Returns an GadgetAssemblage object containing only this gadget. Note that this overwrites any currently specified global grid.
    def wrap_gadget(self):
        pass
        height = max(self.a, self.b)
        full_legs = [(k, 0) for k in range(height)] + [(k, 1) for k in range(height)]
        i0_map = {leg:(leg[0], 0) for leg in full_legs}
        i1_map = {leg:(leg[0], 1) for leg in full_legs}
        i0 = Interlink(height, "i0_%s"%self.label, i0_map)
        i1 = Interlink(height, "i1_%s"%self.label, i1_map)
        g0_map = {leg:leg for leg in self.legs}
        g0 = Gadget(self.a, self.b, self.label, g0_map)
        # Instantiate gadget objects
        g_list = [g0]
        i_list = [i0, i1]
        wrapped_assemblage = GadgetAssemblage(g_list, i_list)
        return wrapped_assemblage

class Interlink:

    def __init__(self, a, label, map_to_grid=dict()):
        self.a = a
        self.label = label
        self.legs = [(k, 0) for k in range(a)] + [(k, 1) for k in range(a)]
        self.map_to_grid = map_to_grid

class GadgetAssemblage:

    def __init__(self, gadgets, interlinks):
        self.gadgets = gadgets
        self.interlinks = interlinks
        self.global_grid = self.instantiate_global_grid()
        self.gadget_set = {g.label:g for g in gadgets}

        # Check that gadget specification is proper.
        is_valid = self.is_valid_instantiation()

        if is_valid:
            # print("Assemblage is valid; generating internal objects.")
            # Instantiate dictionary gadgets and then wires. Note order.
            self.gadget_dict = self.gen_gadget_dict_gadgets()
            self.gen_gadget_dict_wires()
            # Instantiate derived properties of the gadget.
            self.input_legs, self.output_legs, self.input_dict, self.output_dict = self.get_terminal_legs()
            self.shape = (len(self.input_legs), len(self.output_legs))
            self.origin_guide = self.gen_leg_origin_guide()
        else:
            raise NameError('Gadget instantiation failed.')

    #############################################
    #############################################
    #############################################
    #############################################
    """
    BEGIN MODIFIED COMPOSITE_GADGET METHODS
    """

    # Note redefinition of this in separate, named file.
    def get_sequence(self):
        pass

    # This should be identical to Jack's implementation, as it calls get_sequence().
    def get_unitary(self):
        pass 

    # This should be identical to Jack's implementation, as it calls get_sequence().
    def get_qsp_unitary(self):
        pass

    """
    END MODIFIED COMPOSITE_GADGET METHODS
    """
    #############################################
    #############################################
    #############################################
    #############################################

    """
    Given a series of Gadget and Interlink objects, determines if they can, up to the rules specified, be validly recognized as a gadget assemblage. This means that the gadgets' global coordinates are sequential in x, non-overlapping but contiguous in y, and that these global coordinates are increasing with the index in the list.
    """
    def is_valid_instantiation(self):
        gadgets = self.gadgets
        interlinks = self.interlinks

        if (len(interlinks) != (len(gadgets) + 1)) or len(gadgets) == 0:
            raise NameError("len(gadgets) is not len(interlinks) - 1, or len(gadgets) is zero.")
        else:
            x_window = 0
            y_window = (0, 0)

            gadget_label_set = set()
            gadget_label_list = list()

            # Checking gadgets for proper size and spacing.
            for g in gadgets:
                y_window = (y_window[1], y_window[1] + max(g.a, g.b))
                leg_map = g.map_to_grid

                if set(leg_map.keys()) != set(g.legs):
                    diff = list(set(leg_map.keys()).difference(set(g.legs)))
                    raise NameError("Gadget %s has disallowed local legs: %s." % (g.label, str(diff)))
                elif (g.a < 1) or (g.b < 1):
                    raise NameError("Gadget %s has an invalid size: %s." % str((g.a, g.b)))
                else:
                    pass

                gadget_label_set.add(g.label)
                gadget_label_list.append(g.label)

                in_x_set = set()
                out_x_set = set()
                in_y_set = set()
                out_y_set = set()
                in_y_list = []
                out_y_list = []
                y_set = set(range(y_window[0], y_window[1]))
                
                for elem in leg_map.keys():
                    if elem[1] == 0:
                        in_y_set.add(leg_map[elem][0])
                        in_y_list.append(leg_map[elem][0])
                        in_x_set.add(leg_map[elem][1])
                    elif elem[1] == 1:
                        out_y_set.add(leg_map[elem][0])
                        out_y_list.append(leg_map[elem][0])
                        out_x_set.add(leg_map[elem][1])
                    else:
                        raise NameError('Gadget %s local leg %s invalid.' % (g.label, str(elem)))

                x_check = (in_x_set == {x_window}) and (out_x_set == {x_window + 1})
                y_check = in_y_set.issubset(y_set) and out_y_set.issubset(y_set)

                y_in_dup = len(in_y_set) == len(in_y_list)
                y_out_dup = len(out_y_set) == len(out_y_list)
                y_dup_check = y_in_dup and y_out_dup

                if not y_in_dup:
                    raise NameError("Gadget %s duplicate input legs." % g.label)
                elif not y_out_dup:
                    raise NameError("Gadget %s duplicate output legs." % g.label)
                elif not y_check:
                    raise NameError("Gadget %s legs out of y range." % g.label)
                elif not x_check:
                    raise NameError("Gadget %s legs out of x range." % g.label)
                else:
                    pass

                x_window = x_window + 1

            # Perform analogous checks for all interlinks.
            interlink_label_set = set()
            interlink_label_list = list()

            # Checking interlinks for proper size and spacing.
            x_window = 0
            for i in interlinks:
                leg_map = i.map_to_grid

                if set(leg_map.keys()) != set(i.legs):
                    diff = list(set(leg_map.keys()).symmetric_difference(set(i.legs)))
                    raise NameError("Interlink %s has disallowed local legs: %s." % (i.label, str(diff)))
                elif (i.a < 1):
                    raise NameError("Gadget %s has an invalid size: %s." % str(i.a))
                else:
                    pass

                interlink_label_set.add(i.label)
                interlink_label_list.append(i.label)

                in_x_set = set()
                out_x_set = set()
                in_y_set = set()
                out_y_set = set()
                in_y_list = []
                out_y_list = []
                # Note interlinks have no width in global coordinates.
                y_set = set(range(len(self.global_grid)))

                for elem in leg_map.keys():
                    if elem[1] == 0:
                        in_y_set.add(leg_map[elem][0])
                        in_y_list.append(leg_map[elem][0])
                        in_x_set.add(leg_map[elem][1])
                    elif elem[1] == 1:
                        out_y_set.add(leg_map[elem][0])
                        out_y_list.append(leg_map[elem][0])
                        out_x_set.add(leg_map[elem][1])
                    else:
                        raise NameError('Interlink %s local leg %s invalid.' % (i.label, str(elem)))
                
                x_check = (in_x_set == {x_window}) and (out_x_set == {x_window})
                y_check = in_y_set.issubset(y_set) and out_y_set.issubset(y_set)

                y_in_dup = len(in_y_set) == len(in_y_list)
                y_out_dup = len(out_y_set) == len(out_y_list)
                y_dup_check = y_in_dup and y_out_dup

                if not y_in_dup:
                    raise NameError("Interlink %s duplicate input legs." % i.label)
                elif not y_out_dup:
                    raise NameError("Interlink %s duplicate output legs." % i.label)
                elif not y_check:
                    raise NameError("Interlink %s legs out of y range." % i.label)
                elif not x_check:
                    raise NameError("Interlink %s legs out of x range." % i.label)
                else:
                    pass

                x_window = x_window + 1

            # Checking ghost wire compatibility
            ghost_check, ghost_check_data = self.check_interlink_ghost_legs()
            # Checking uniqueness of labels
            gadget_name_check = len(gadget_label_set) == len(gadget_label_list)
            interlink_name_check = len(interlink_label_set) == len(interlink_label_list)
            # Forbidding privileged name for wires.
            gadget_name_safe = not("WIRE" in gadget_label_set)
            
            if not ghost_check:
                raise NameError("Ghost legs permuted improperly.")
            elif not gadget_name_check:
                raise NameError("Multiple gadgets with same name.")
            elif not interlink_name_check:
                raise NameError("Multiple interlinks with same name.")
            elif not gadget_name_safe:
                raise NameError("A gadget improperly named 'WIRE'.")
            else:
                pass
            
            return True

    def get_ghost_legs(self):
        ghost_legs_f = [[] for k in range(len(self.global_grid[0]))]
        ghost_legs_r = [[] for k in range(len(self.global_grid[0]))]

        # Dictionary mapping legs to their gadgets.
        ghost_legs_f_map = dict()
        ghost_legs_r_map = dict()

        # Forward pass
        for k in range(len(self.global_grid[0]) - 1):
            current_tally = []
            g = self.gadgets[k]
            a, b = g.a, g.b
            if b < a:
                # Note mapping toward left, as we don't want permutation
                local_ghost_legs = [(j, 0) for j in range(b, a)]
                ghost_y = list(map(lambda x: g.map_to_grid[x][0], local_ghost_legs))
                current_tally = current_tally + ghost_y
                for y in ghost_y:
                    ghost_legs_f_map[y] = g.label
            else:
                pass
            
            # Keep tally of previous ghost legs as well; not starts loading at 1
            replace_item = ghost_legs_f[k+1] + current_tally + ghost_legs_f[k]
            ghost_legs_f[k+1] = replace_item.copy()

        # Backward pass
        for k in range(len(self.global_grid[0]) - 1):
            current_tally = []
            reverse_index = len(self.global_grid[0]) - 1 - k
            g = self.gadgets[reverse_index - 1]
            a, b = g.a, g.b
            if a < b:
                # Note mapping toward left, as we don't want permutation
                local_ghost_legs = [(j, 1) for j in range(a, b)]
                # Note that map_to_grid doesn't go through interlink to right
                ghost_y = list(map(lambda x: g.map_to_grid[x][0], local_ghost_legs))
                current_tally = current_tally + ghost_y
                for y in ghost_y:
                    ghost_legs_r_map[y] = g.label
            else:
                pass
            # Keep tally of previous ghost legs as well
            replace_item = ghost_legs_r[reverse_index-1] + current_tally + ghost_legs_r[reverse_index]
            ghost_legs_r[reverse_index-1] = replace_item.copy()

        ghost_legs = [ghost_legs_f[k] + ghost_legs_r[k] for k in range(len(self.global_grid[0]))]
        
        # Return overall list, and reverse and forward passes
        return (ghost_legs, ghost_legs_f_map, ghost_legs_r_map)

    def check_interlink_ghost_legs(self):

        ghost_legs = self.get_ghost_legs()[0]
        interlinks = self.interlinks
        is_valid = True 
        check_list = []

        for k in range(len(interlinks)):
            ghost_permutation = list(map(lambda x: interlinks[k].map_to_grid[(x,1)][0], ghost_legs[k]))
            if set(ghost_legs[k]) == set(ghost_permutation):
                check_list.append(True)
            else:
                check_list.append(False)
                is_valid = False
        # Returns validity and a check of each interlink.
        return (is_valid, check_list)

    def instantiate_global_grid(self):
        grid_len_0 = sum([max(g.a, g.b) for g in self.gadgets])
        grid_len_1 = len(self.gadgets) + 1
        global_grid = [[(j, k) for k in range(grid_len_1)] for j in range(grid_len_0)]
        return global_grid

    """
    A quick way to print a gadget using gadget_dict to a tikz-expressible form; currently contiguous lines are not automatically combined, which may yield a less pleasing plot.
    """
    def print_assemblage(self):
        # At each point, draw a wire to the relevant end points.
        print_str = ""

        paths = []
        for y in range(len(self.global_grid)):
            for x in range(len(self.global_grid[0])):
                neighbors = self.gadget_dict[(y, x)]
                if neighbors == (None, None):
                    continue
                else:
                    # Retrieve left and right neighbors.
                    nl = neighbors[0]
                    nr = neighbors[1]
                    l_coord_list = []

                    # Check if gadget to the left.
                    if nl[0] != "WIRE":
                        lcoord = self.gadget_set[nl[0]].map_to_grid[nl[2]]
                        l_coord_list.append((lcoord[0], lcoord[1] - 0.75))
                        l_coord_list.append((lcoord[0], lcoord[1] - 0.5))
                        l_coord_list.append((y, x))
                        # Check if leads into a wire, and extend if so.
                        if nr[0] == "WIRE":
                            l_coord_list.append((y, x + 0.5))
                        else:
                            l_coord_list.append((y, x + 0.25))
                    else:
                        if nl[2] == "TERMINAL":
                            lcoord = nl[1]
                            l_coord_list.append((lcoord[0], lcoord[1]))
                            l_coord_list.append((lcoord[0], lcoord[1] + 0.5))
                            l_coord_list.append((y, x))
                            if nr[0] != "WIRE":
                                l_coord_list.append((y, x + 0.25))
                            else:
                                l_coord_list.append((y, x + 0.5))
                        else:
                            if nr[0] == "WIRE":
                                lcoord = nl[1]
                                l_coord_list.append((lcoord[0], lcoord[1] + 0.5))
                                l_coord_list.append((y, x))
                                l_coord_list.append((y, x + 0.5))
                            else:
                                lcoord = nl[1]
                                l_coord_list.append((lcoord[0], lcoord[1] + 0.5))
                                l_coord_list.append((y, x))
                                l_coord_list.append((y, x + 0.25))

                    for k in range(len(l_coord_list)):
                        l_coord_list[k] = (l_coord_list[k][0], l_coord_list[k][1]*2.5)

                    paths.append(list(map(lambda x: x[::-1], l_coord_list)))

        color_list = ["Apricot", "Aquamarine", "Bittersweet", "Black", "Blue", "BlueGreen", "BlueViolet", "BrickRed", "Brown", "BurntOrange", "CadetBlue", "CarnationPink", "Cerulean", "CornflowerBlue", "Cyan", "Dandelion", "DarkOrchid", "Emerald", "ForestGreen", "Fuchsia", "Goldenrod", "Gray", "Green", "GreenYellow", "JungleGreen", "Lavender", "LimeGreen", "Magenta", "Mahogany", "Maroon", "Melon", "MidnightBlue", "Mulberry", "NavyBlue", "OliveGreen", "Orange", "OrangeRed", "Orchid", "Peach", "Periwinkle", "PineGreen", "Plum", "ProcessBlue", "Purple", "RawSienna", "Red", "RedOrange", "RedViolet", "Rhodamine", "RoyalBlue", "RoyalPurple", "RubineRed", "Salmon", "SeaGreen", "Sepia", "SkyBlue", "SpringGreen", "Tan", "TealBlue", "Thistle", "Turquoise", "Violet", "VioletRed", "WildStrawberry", "Yellow", "YellowGreen", "YellowOrange"]

        # Use of path joining subroutine to make things neater
        combined_paths = self.combine_paths(paths)

        index = 0
        for path in combined_paths:
            str_path = "--".join(list(map(lambda x: str(x), path)))
            color = color_list[(13*index)%len(color_list)]
            index = index + 1
            print_str = print_str + ("\\draw[line width=0.18cm, color=white] %s;\n" % (str_path))
            print_str = print_str + ("\\draw[color=%s] %s;\n" % (color, str_path))
                    
        for g in self.gadgets:
            size = max(g.a, g.b)
            top_coord = g.map_to_grid[(0, 0)]
            bottom_coord = (top_coord[0] + size, top_coord[1] + 0.5)

            top_coord = top_coord[::-1]
            bottom_coord = bottom_coord[::-1]

            top_coord = (2.5*(top_coord[0]+0.1), 1*(top_coord[1]-0.5))
            bottom_coord = (2.5*(bottom_coord[0]-0.1), 1*(bottom_coord[1]-0.5))

            label_coord = ((top_coord[0] + bottom_coord[0])/2, (top_coord[1] + bottom_coord[1])/2)

            print_str = print_str + ("\\draw[fill=white] %s rectangle %s;\n" % (top_coord, bottom_coord))
            print_str = print_str + ("\\node[anchor=center] at %s {\\large %s};\n" % (label_coord, g.label))

        for leg in self.input_legs:
            true_leg = self.reverse_interlink(self.interlinks[0])[(leg, 1)][0]
            print_str = print_str + ("\\node[anchor=east] at %s {\\large $x_{%s}$};\n" % ((-2.5, true_leg), str(true_leg)))

        for leg in self.output_legs:
            length = 2.5*(len(self.global_grid[0])) - 1.25
            print_str = print_str + ("\\node[anchor=west] at %s {\\large $y_{%s}$};\n" % ((length, leg), str(leg)))

        return print_str

    """
    A pleasant helper function for combining lists with identical endpoints; this is used in print_assemblage, to simplify the form of the resulting diagram, and allow for easier tex styling.
    """
    def combine_paths(self, paths):
        
        # This algorithm assumes that paths cannot form loops
        local_paths = paths.copy()

        while True:
            new_local_paths = []
            end_dict = {local_paths[k][-1]:k for k in range(len(local_paths))}
            start_dict = {local_paths[k][0]:k for k in range(len(local_paths))}
            possible_connections = []
            seen_set = set()
            for elem in end_dict.keys():
                if elem in start_dict and not (end_dict[elem] in seen_set) and not (start_dict[elem] in seen_set):
                    possible_connections.append((end_dict[elem], start_dict[elem]))
                    seen_set.add(end_dict[elem])
                    seen_set.add(start_dict[elem])
            if len(possible_connections) > 0:
                for elem in possible_connections:
                    new_path = local_paths[elem[0]] + local_paths[elem[1]][1:]
                    new_local_paths.append(new_path)
                remaining = list(set(range(len(local_paths))) - seen_set)
                new_local_paths = new_local_paths + [local_paths[k] for k in remaining]
                # Finally, replace and iterate
                local_paths = new_local_paths
            else:
                break
        return local_paths

    """
    TODO: we need to do a reverse pass in this method; in the case of, for instance, a 2, 1 gadget, the current method would think that both output legs are terminal, because we can't get there from an input leg. But we see this if we go from the end. Anything that eventually reaches a gadget is a terminal leg.
    """
    def get_terminal_legs(self):
        input_legs = []
        output_legs = []

        input_leg_dict = dict()
        output_leg_dict = dict()

        ghost_legs = self.get_ghost_legs()[0]
        initial_ghost_legs = ghost_legs[0]
        final_ghost_legs = ghost_legs[-1]

        # Forward pass.
        for y in range(len(self.global_grid)):
            if y in initial_ghost_legs:
                continue
            else:
                is_input = False
                tracked_y = y
                true_leg = self.interlinks[0].map_to_grid[(y, 1)][0]
                for x in range(len(self.global_grid[0])):
                    # Query correct interlink.
                    interlink = self.interlinks[x]
                    # Follow the interlink based on current y position.
                    tracked_y = interlink.map_to_grid[(tracked_y, 1)][0]
                    if len(self.get_gadget(x, tracked_y, 1)) != 0:
                        is_input = True # Leg is terminal if it sees a gadget.
                        input_legs.append(true_leg)
                        g_name = (self.get_gadget(x, tracked_y, 1)[0],)
                        g_leg = self.gadget_dict[(tracked_y, x)][1][2]
                        input_leg_dict[g_name + g_leg] = true_leg
                        break
        # Backwards pass.
        for y in range(len(self.global_grid)):
            if y in final_ghost_legs:
                continue
            else:
                is_output = False
                tracked_y = y
                for x in range(len(self.global_grid[0])):
                    rev_x = len(self.global_grid[0]) - 1 - x
                    # Query correct interlink.
                    interlink = self.interlinks[rev_x]
                    # check for gadget before following interlink
                    if len(self.get_gadget(rev_x, tracked_y, 0)) != 0:
                        is_output = True # Leg is terminal if it sees a gadget.
                        output_legs.append(y) # Append original leg for input.
                        g_name = (self.get_gadget(rev_x, tracked_y, 0)[0],)
                        g_leg = self.gadget_dict[(tracked_y, rev_x)][0][2]
                        output_leg_dict[g_name + g_leg] = y
                        break
                    # Follow the interlink based on current y position (backwards), but after gadget check!
                    tracked_y = self.reverse_interlink(interlink)[(tracked_y, 1)][0]                
        return (sorted(input_legs), sorted(output_legs), input_leg_dict, output_leg_dict)

    def get_gadget(self, x, y, lr):

        # Check right (lr = 1) or left (lr = 0) for gadget name, if it exists.
        linked_gadgets = self.gadget_dict[(y, x)]
        if (linked_gadgets[lr] != None) and (linked_gadgets[lr][0] != "WIRE"):
            return [linked_gadgets[lr][0]]
        else:
            return []

    """
    Returns an interlink map-to-grid object applying the inverse permutation of the one supplied in the argument, with respect to the same global grid points.
    """
    def reverse_interlink(self, interlink):
        reverse_map_to_grid = dict()
        list_in = []
        list_out = []
        
        for k in range(interlink.a):
            list_in.append(interlink.map_to_grid[(k, 0)])
            list_out.append(interlink.map_to_grid[(k, 1)])

        for k in range(interlink.a):
            reverse_map_to_grid[(k, 1)] = list_in[list_out.index(list_in[k])]
            reverse_map_to_grid[(k, 0)] = list_out[list_in.index(list_out[k])]

        return reverse_map_to_grid

    """
    Note extended terminal coordinates, and specific names for wires (WIRE), which should be protected in the instantiation method.
    """
    def gen_gadget_dict_wires(self):
        
        input_legs, output_legs, i_dict, o_dict = self.get_terminal_legs()
        global_input_legs = list(map(lambda x: (x, 0), input_legs))
        global_output_legs = list(map(lambda x: (x, len(self.global_grid[0]) - 1), output_legs))

        current_check = global_input_legs

        # Flood through assemblage from known initial legs.
        while len(current_check) > 0:
            new_check = []
            for elem in current_check:
                # Check if at beginning of list.
                if elem[1] == 0:
                    # Choose first interlink.
                    interlink = self.interlinks[0]
                    # Compute where interlink leads.
                    following_step = self.interlinks[1].map_to_grid[elem[0], 1]
                    previous_step = self.reverse_interlink(self.interlinks[0])[elem[0], 1]
                    previous_step = (previous_step[0], previous_step[1] - 1)
                    # Check gadgets connecting to where interlink led.
                    possible_gadgets = self.gadget_dict[elem]
                    # If there is None to right.
                    if possible_gadgets[1] == None:
                        # Assign terminal wire to next_step location.
                        assignment = (("WIRE", previous_step, "TERMINAL"), ("WIRE", following_step, "INTERNAL"))
                        # Follow wire and add end to new check list.
                        new_check.append(following_step)
                    else:
                        # Assign terminal wire to next_step location.
                        assignment = (("WIRE", previous_step, "TERMINAL"), self.gadget_dict[elem][1])
                        # Locate gadget by its name.
                        g = self.gadget_set[self.gadget_dict[elem][1][0]]
                        # Compute local outgoing legs of the gadget.
                        gadget_out_legs = filter(lambda x: x[1] == 1, g.legs)
                        # Compute global outgoing legs of the gadget.
                        gadget_global_out_legs = list(map(lambda x: g.map_to_grid[x], gadget_out_legs))
                        # Permute out legs by next permutation.
                        gadget_global_out_legs = list(map(lambda x: self.interlinks[1].map_to_grid[(x[0], 1)], gadget_global_out_legs))
                        # Add all global outgoing legs to new check list.
                        new_check = new_check + gadget_global_out_legs
                    # Finally, assign new tuple to next_step location.

                    self.gadget_dict[elem] = assignment
                # Check if at the end of the list.
                elif elem[1] == len(self.global_grid[0]) - 1:
                    extended_elem = (elem[0], elem[1] + 1)
                    # Choose the last interlink.
                    interlink = self.interlinks[elem[1]]
                    # Compute where the interlink leads (globally).
                    previous_step = self.reverse_interlink(self.interlinks[elem[1]])[elem[0], 1]
                    previous_step = (previous_step[0], previous_step[1] - 1)
                    # Query possible gadgets.
                    possible_gadgets = self.gadget_dict[elem]
                    # If there is WIRE or gadget to the left.
                    if (possible_gadgets[0] != None):
                        # If there is a gadget to the left.
                        if (possible_gadgets[0][0] != "WIRE"):
                            assignment = (possible_gadgets[0], ("WIRE", extended_elem, "TERMINAL"))
                        else:
                            assignment = (("WIRE", previous_step, "INTERNAL"), ("WIRE", extended_elem, "TERMINAL"))
                    # If there is None to the left.
                    else:
                        # BUG: check to make sure this is always terminal
                        assignment = (("WIRE", previous_step, "INTERNAL"), ("WIRE", extended_elem, "TERMINAL"))
                    # Finally, assign new tuple to next_step location.
                    self.gadget_dict[elem] = assignment
                # If not at the end or beginning of the list.
                else:
                    # Find the x coordinate and corresponding interlink.
                    x_coord = elem[1]
                    interlink = self.interlinks[x_coord]
                    # Compute where the interlink leads (globally).
                    following_step = self.interlinks[x_coord + 1].map_to_grid[elem[0], 1]
                    previous_step = self.reverse_interlink(self.interlinks[x_coord])[elem[0], 1]
                    previous_step = (previous_step[0], previous_step[1] - 1)
                    # Compute possible gadgets.
                    possible_gadgets = self.gadget_dict[elem]
                    # If there is no gadget or WIRE to the right.
                    if self.gadget_dict[elem][1] == None:
                        # If also no gadget or WIRE to the left.
                        if self.gadget_dict[elem][0] == None:
                            # Assign wire at next step to where it comes from and where it leads.
                            assignment = (("WIRE", previous_step, "INTERNAL"), ("WIRE", following_step, "INTERNAL"))
                            # Finally, assign the next step to the new_check.
                            new_check.append(following_step)
                        # If there is a WIRE to the left.
                        elif self.gadget_dict[elem][0][0] == "WIRE":
                            # Assign wire at next step to where it comes from and where it leads.
                            assignment = (("WIRE", previous_step, "INTERNAL"), ("WIRE", following_step, "INTERNAL"))
                            # Finally, assign the next step to the new_check.
                            new_check.append(following_step)
                        # If there is a gadget to the left.
                        else:
                            # Look ahead to see where the wire goes.
                            assignment = (self.gadget_dict[elem][0], ("WIRE", following_step, "INTERNAL"))
                            # Finally, assign the next step to the new_check.
                            new_check.append(following_step)
                    # If there is a gadget to the right (note there will never be a WIRE to the right).
                    else:
                        # If there is None to the left.
                        if self.gadget_dict[elem][0] == None:
                            assignment = (("WIRE", previous_step, "INTERNAL"), self.gadget_dict[elem][1])
                        # If there is a WIRE to the left (this never occurs).
                        elif self.gadget_dict[elem][0][0] == "WIRE":
                            assignment = (("WIRE", previous_step, "INTERNAL"), self.gadget_dict[elem][1])
                        # If there is a gadget to the left.
                        else:
                            # This operation is trivial, but good for completeness.
                            assignment = (self.gadget_dict[elem][0], self.gadget_dict[elem][1])
                        # Call the gadget by its name.
                        g = self.gadget_set[self.gadget_dict[elem][1][0]]
                        # Compute the outgoing legs of the gadget (local).
                        gadget_out_legs = filter(lambda x: x[1] == 1, g.legs)
                        # Compute the outgoing legs of the gadget (global).
                        gadget_global_out_legs = list(map(lambda x: g.map_to_grid[x], gadget_out_legs))
                        # For a given position, gadget legs are assigned to where the interlink takes them before a given x coordinate.
                        gadget_global_out_legs = list(map(lambda x: self.interlinks[x_coord + 1].map_to_grid[(x[0],1)], gadget_global_out_legs))
                        # Add all the new legs to the check list.
                        new_check = new_check + gadget_global_out_legs
                    # Finally, assign the result proper location in gadget_dict.
                    self.gadget_dict[elem] = assignment
            # Remove duplicates in new_check, and fill queue for next iteration.
            current_check = list(set(new_check))

    """
    Given a valid series of gadgets and interlinks, modifies a _blank_ gadget_dict such that grid points which connect to a gadget leg (through an interlink to the left, or through nothing to the right) are marked with a tuple element (gadget_name, global_coord, local_coord) at dictionary key global_coord. Each global coord maps to a tuple of possibly two such elements, representing the left and right neighbors of global_coord. Wire notation is handled by gen_gadget_dict_wires, to be run only after gen_gadget_dict_gadgets.
    """
    def gen_gadget_dict_gadgets(self):
        gadget_dict = dict()
        for j in range(len(self.global_grid)):
            for k in range(len(self.global_grid[0])):
                gadget_dict.setdefault((j, k), (None, None))

        for g in self.gadgets:
            local_coords = g.map_to_grid
            left_local = list(filter(lambda x: x[1] == 0, local_coords.keys()))
            right_local = list(filter(lambda x: x[1] == 1, local_coords.keys()))
            left_global = list(map(lambda x: local_coords[x], left_local))
            right_global = list(map(lambda x: local_coords[x], right_local))

            left_local_coords = left_local.copy()
            right_local_coords = right_local.copy()
            global_to_local_left = {g.map_to_grid[x]:x for x in left_local_coords}
            global_to_local_right = {g.map_to_grid[x]:x for x in right_local_coords}
            
            # Find the relevant interlink.
            g_x_coord = g.map_to_grid[(0,1)][1]
            right_interlink = self.interlinks[g_x_coord]

            # Look at the coord and add a right neighbor if we have a left global, and a left neighbor if we have a right global. Otherwise, we may just have a WIRE (handled later), or nothing.
            for elem in left_global:
                arg0, arg1 = gadget_dict[elem]
                gadget_dict[elem] = (arg0, (g.label, elem, global_to_local_left[elem]))
            for elem in right_global:
                interlink_y_coord = elem[0]
                
                # Important note: the gadget_dict looks from a point through an interlink for the outgoing legs of gadgets, but only to the left.
                permuted_elem = right_interlink.map_to_grid[(interlink_y_coord,1)]
                # Unwrap existing elements.
                arg0, arg1 = gadget_dict[elem]
                # Instantiate dictionary entry.
                gadget_dict[permuted_elem] = ((g.label, permuted_elem, global_to_local_right[elem]), arg1)
        return gadget_dict

    def get_component_ends(self, gadgets):
        min_index = min([g.map_to_grid[(0,0)][1] for g in gadgets])
        max_index = max([g.map_to_grid[(0,1)][1] for g in gadgets])
        return (min_index, max_index)

    def is_valid_component(self, gadgets):
        """
        Currently this is only used and valid for adjacent gadgets; however, we can use it for the full case, not currently defined. We should check for adjacency, and also allow this to be called with gadget names rather than objects, with a check beforehand.
        """

        # Trivial lists are not valid
        if len(gadgets) == 0 or len(gadgets) > 2:
            raise NameError("Cannot call is_valid_component with no gadgets or more than two gadgets.")
        else:
            if len(gadgets) == 2:
                index_0 = self.gadget_set[gadgets[0].label].map_to_grid[(0,0)][1]
                index_1 = self.gadget_set[gadgets[1].label].map_to_grid[(0,0)][1]
                if abs(index_0 - index_1) != 1:
                    raise NameError("Cannot call is_valid_component on two gadgets if they are not adjacent.")
            else:
                pass

            g_name_set = {g.label for g in gadgets}
            seen_set = set()

            min_index, max_index = self.get_component_ends(gadgets)

            # define input and output legs
            input_legs = []
            output_legs = []

            # Initialize with all output legs of current gadgets
            check_list = []
            for g in gadgets:
                interlink_index = g.map_to_grid[(0,1)][1]
                interlink = self.interlinks[interlink_index]
                for elem in g.legs:
                    if elem[1] == 0:
                        maps_to = g.map_to_grid[elem]
                    else:
                        maps_to = interlink.map_to_grid[(g.map_to_grid[elem][0], 1)]
                    check_list.append(maps_to)

            # Add all initial nodes to seen_set
            seen_set.update(check_list)

            # Start with the assumption of validity.
            is_valid = True

            # While still elements to check
            while(len(check_list) != 0):
                new_check_list = []
                for elem in check_list:
                    if elem[1] == min_index:
                        input_legs.append(elem[0])
                    elif elem[1] == max_index:
                        output_legs.append(elem[0])
                    else:
                        pass
                    # Check if at ends, and otherwise continue
                    if (elem[1] == min_index) or (elem[1] == max_index):
                        continue
                    else:
                        neighbors = self.gadget_dict[(elem)]
                        # If none, then pass (should never occur).
                        if neighbors == (None, None):
                            continue
                        else:
                            # Wire to the left
                            if neighbors[0][0] == "WIRE":
                                # Check if an output leg
                                if neighbors[0][1][1] == min_index:
                                    input_legs.append(neighbors[0][1][0])
                                elif neighbors[0][1][1] == max_index:
                                    output_legs.append(neighbors[0][1][0])
                                else:
                                    pass
                                if neighbors[0][0] == "TERMINAL":
                                    continue
                                else:
                                    if (neighbors[0][1][1] <= min_index) or (neighbors[0][1][1] >= max_index):
                                        continue
                                    else:
                                        if not(neighbors[0][1] in seen_set):
                                            new_check_list.append(neighbors[0][1])
                            # Gadget to the left
                            else:
                                g_name = neighbors[0][0]
                                if not(g_name in g_name_set):
                                    is_valid = False
                                    return (is_valid, None)

                            # Wire to the right
                            if neighbors[1][0] == "WIRE":
                                # Check if an output leg
                                if neighbors[1][1][1] == min_index:
                                    input_legs.append(neighbors[1][1][0])
                                elif neighbors[1][1][1] == max_index:
                                    output_legs.append(neighbors[1][1][0])
                                else:
                                    pass
                                if neighbors[1][0] == "TERMINAL":
                                    continue
                                else:
                                    if (neighbors[1][1][1] <= min_index) or (neighbors[1][1][1] >= max_index):
                                        continue
                                    else:
                                        if not(neighbors[1][1] in seen_set):
                                            new_check_list.append(neighbors[1][1])
                            # Gadget to the right
                            else:
                                g_name = neighbors[1][0]
                                if not(g_name in g_name_set):
                                    is_valid = False
                                    return (is_valid, None)
                check_list = new_check_list
                seen_set.update(check_list)
            # Finally, return true if broken from loop
            if is_valid:
                return (is_valid, list(set(input_legs)), list(set(output_legs)))
            else:
                return (is_valid, None)

    """
    Performs the same as swap_adjacent_gadgets(), but allows one to call gadgets by name.
    """
    def swap_gadgets(self, g0, g1):
        
        if not (g0 in self.gadget_set) or not (g1 in self.gadget_set):
            raise NameError("Gadget %s or %s does not exist" % (g0, g1))
        else:
            gadget_0 = self.gadget_set[g0]
            gadget_1 = self.gadget_set[g1]
            return self.swap_adjacent_gadgets(gadget_0, gadget_1)

    """
    This method updates the local maps of two adjacent gadgets which are not part of the same component; it should be able to do this without a call to the gadget_dict; if not, then it should update everything including the dict; we can verify by plotting.
    """
    def swap_adjacent_gadgets(self, gadget_0, gadget_1):
        # we can check g0 and g1 as a paired component if we want to be really sure.
        is_valid_0 = self.is_valid_component([gadget_0])
        is_valid_1 = self.is_valid_component([gadget_1])

        comp_ends_0 = self.get_component_ends([gadget_0])
        comp_ends_1 = self.get_component_ends([gadget_1])
        
        # Should be validated that these are adjacent
        i0_index = comp_ends_0[0]
        i1_index = comp_ends_0[0] + 1
        i2_index = comp_ends_0[0] + 2

        new_g0_map = dict() # g0 map to grid
        new_g1_map = dict() # g1 map to grid
        new_i0_map = dict() # before interlink
        new_i1_map = dict() # middle interlink
        new_i2_map = dict() # after interlink

        # Legs to the left out of g0
        ghost_legs_f_0 = self.get_ghost_legs()[2]
        ghost_legs_f_0 = list(filter(lambda x: ghost_legs_f_0[x] == gadget_0.label, ghost_legs_f_0.keys()))
        # Legs to the left out of g1
        ghost_legs_f_1 = self.get_ghost_legs()[2]
        ghost_legs_f_1 = list(filter(lambda x: ghost_legs_f_1[x] == gadget_1.label, ghost_legs_f_1.keys()))
        # Legs to the right out of g0
        ghost_legs_f_2 = self.get_ghost_legs()[1]
        ghost_legs_f_2 = list(filter(lambda x: ghost_legs_f_2[x] == gadget_0.label, ghost_legs_f_2.keys()))
        # Legs to the right out of g1
        ghost_legs_f_3 = self.get_ghost_legs()[1]
        ghost_legs_f_3 = list(filter(lambda x: ghost_legs_f_3[x] == gadget_1.label, ghost_legs_f_3.keys()))

        ghost_legs_total = ghost_legs_f_0 + ghost_legs_f_1 + ghost_legs_f_2 + ghost_legs_f_3

        # Gadget swap is simple if adjacent, just up and down by heights
        for elem in gadget_0.map_to_grid.keys():
            shift = max(gadget_1.a, gadget_1.b)
            map_0 = gadget_0.map_to_grid[elem][0] + shift
            map_1 = gadget_0.map_to_grid[elem][1] + 1
            new_g0_map[elem] = (map_0, map_1)

        for elem in gadget_1.map_to_grid.keys():
            shift = max(gadget_0.a, gadget_0.b)
            map_0 = gadget_1.map_to_grid[elem][0] - shift
            map_1 = gadget_1.map_to_grid[elem][1] - 1
            new_g1_map[elem] = (map_0, map_1)


        """
        *
        *
        * First step: scooting interlinks out of g0 and into g1 to the side.
        *
        *
        """


        g1_input_legs = is_valid_1[1] # TODO: need to account for ghost legs here; this will be relevant later!

        first_interlink = self.interlinks[i0_index]
        middle_interlink = self.interlinks[i1_index]
        pre_images_0 = [self.reverse_interlink(first_interlink)[(self.reverse_interlink(middle_interlink)[(elem, 1)][0], 1)][0] for elem in g1_input_legs]

        change_list_0 = dict()
        for elem in (g1_input_legs + ghost_legs_f_1): # BUG: added ghost legs
            # Find where input legs of g1 map to
            pre_image = self.reverse_interlink(middle_interlink)[(elem, 1)][0]
            pre_pre_image = self.reverse_interlink(first_interlink)[(pre_image, 1)][0]
            change_list_0[(pre_pre_image, 1)] = (elem, i0_index)
            # Solve collision
            collision_pre_image = self.reverse_interlink(first_interlink)[(elem, 1)][0]
            collision_image = first_interlink.map_to_grid[(pre_pre_image, 1)][0]
            if not(collision_pre_image in pre_images_0):
                change_list_0[(collision_pre_image, 1)] = (collision_image, i0_index)

        # Update changes found.
        first_interlink_map = self.interlinks[i0_index].map_to_grid
        for elem in first_interlink_map.keys():
            if elem[1] == 0:
                # If on left of interlink, keep the same.
                new_i0_map[elem] = first_interlink_map[elem]
            else:
                # If we have a new map, we change the corresponding entry
                if elem in change_list_0.keys():
                    new_i0_map[elem] = change_list_0[elem]
                else:
                    new_i0_map[elem] = first_interlink_map[elem]

        # print("CHANGES FOR LINK 0")
        # for elem in change_list_0.keys():
        #     print(elem, end=" to ")
        #     print(change_list_0[elem])

        # print("ORIGINAL LINK 0")
        # for elem in first_interlink_map.keys():
        #     print(elem, end=" to ")
        #     print(first_interlink_map[elem])

        # print("FIRST CHANGE LINK 0")
        # for elem in new_i0_map.keys():
        #     print(elem, end=" to ")
        #     print(new_i0_map[elem])

        g0_output_legs = is_valid_0[2] # ghost leg including?
        change_list_1 = dict() # contains keys of first interlink to change
        for elem in (g0_output_legs + ghost_legs_f_2):
            # Get relevant interlinks.
            middle_interlink = self.interlinks[i1_index]
            last_interlink = self.interlinks[i2_index]
            next_image = last_interlink.map_to_grid[(elem, 1)][0]
            pre_image = self.reverse_interlink(middle_interlink)[(elem, 1)][0]
            # We specify the entire swap at once
            change_list_1[(pre_image, 1)] = (next_image, i2_index)

            # If pre image is not in the set of elems, then we need to fix the interlink somehow
            collision_pre_image = self.reverse_interlink(last_interlink)[(next_image, 1)][0]

            if collision_pre_image != pre_image: # BUG: should this be true equality, or set membership?
                collision_image = last_interlink.map_to_grid[(pre_image, 1)][0]
                # print("COLLISION")
                # print(collision_pre_image)
                # print(collision_image)
                # print(pre_image)
                change_list_1[(collision_pre_image, 1)] = (collision_image, i2_index) 

        # Update changes found. BUG: WHY CANT WE JUST ALTER second row?
        last_interlink_map = self.interlinks[i2_index].map_to_grid

        for elem in last_interlink_map.keys():
            if elem[1] == 0:
                # If on right of interlink, keep the same.
                new_i2_map[elem] = last_interlink_map[elem]
            else:
                # We are dropping keys here
                if elem in change_list_1.keys():
                    new_i2_map[elem] = change_list_1[elem] # BUG IF DROPPED LEGS (check instantiation of check_list)
                else:
                    new_i2_map[elem] = last_interlink_map[elem] # BUG IF DROPPED LEGS (check instantiation of check_list)

        # print("CHANGE LIST FOR LINK 2")
        # for elem in change_list_1.keys():
        #     print(elem, end=" to ")
        #     print(change_list_1[elem])

        # print("ORIGINAL LINK 2")
        # for elem in last_interlink_map.keys():
        #     print(elem, end=" to ")
        #     print(last_interlink_map[elem])

        # print("FIRST CHANGE FOR LINK 2")
        # for elem in new_i2_map.keys():
        #     print(elem, end=" to ")
        #     print(new_i2_map[elem])


        """
        *
        *
        * Second step: moving interlinks out of g0 and into g1 up and down.
        *
        *
        """

        # Uniformly shifting interlinks
        temp_i0_interlink = Interlink(self.interlinks[i0_index].a, self.interlinks[i0_index].label, new_i0_map)
        temp_reverse_map_0 = self.reverse_interlink(temp_i0_interlink)
        g0_in = is_valid_0[1] + ghost_legs_f_0
        g1_in = is_valid_1[1] + ghost_legs_f_1
        # g1_in = [temp_reverse_map_0[(elem, 1)][0] for elem in g1_in] # Note through interlink

        all_pre_images_0 = [temp_reverse_map_0[(elem, 1)][0] for elem in (g0_in + g1_in)]

        # Shift one set of legs up and another down: BUG SHIFT GHOSTS
        i0_shift_map = dict()
        for elem in new_i0_map.keys():
            if elem[1] == 0:
                # If on left of interlink, keep the same.
                i0_shift_map[elem] = new_i0_map[elem]
            else:
                if new_i0_map[elem][0] in (g0_in):
                    shift = max(gadget_1.a, gadget_1.b)
                    i0_shift_map[elem] = (new_i0_map[elem][0] + shift, new_i0_map[elem][1])
                elif new_i0_map[elem][0] in (g1_in):
                    shift = max(gadget_0.a, gadget_0.b)
                    i0_shift_map[elem] = (new_i0_map[elem][0] - shift, new_i0_map[elem][1])
                else:
                    if not(elem[0] in all_pre_images_0):
                        i0_shift_map[(elem[0], 1)] = new_i0_map[(elem[0], 1)]
                    else:
                        continue

        """
        Something here is causing a loss of the transposition expected on the final row; if we can understand this, then we may understand more about the leg issue (though this is probably from the way that get-terminal legs works)
        """

        # print("SHIFT 0 LEGS")
        # print(g0_in)
        # print(g1_in)
        # print(all_pre_images_0)

        # print("LINK 0 BEFORE shifting")
        # for elem in new_i0_map.keys():
        #     print(elem, end=" to ")
        #     print(new_i0_map[elem])

        # print("LINK 0 after shifting")
        # for elem in i0_shift_map.keys():
        #     print(elem, end=" to ")
        #     print(i0_shift_map[elem])


        temp_i2_interlink = Interlink(self.interlinks[i2_index].a, self.interlinks[i2_index].label, new_i2_map)
        temp_reverse_map_2 = self.reverse_interlink(temp_i2_interlink)
        g0_out = is_valid_0[2] + ghost_legs_f_2
        g1_out = is_valid_1[2] + ghost_legs_f_3
        # NOTE: Possible bug in this code; still not sure how this check is being done.
        g0_out = [self.interlinks[i2_index].map_to_grid[(elem, 1)][0] for elem in g0_out]

        all_pre_images_2 = [temp_reverse_map_2[(elem, 1)][0] for elem in (g0_out + g1_out)]

        """
        In this and the shifting before, we have to check if we map things outside the range of the legs, and repair any breaks we may make. This is not a problem for the first shift, but it is a problem for us generally.
        """
        i2_shift_map = dict()
        for elem in new_i2_map.keys():
            if elem[1] == 0:
                i2_shift_map[elem] = new_i2_map[elem]
            else:
                if elem[0] in (g0_out): # note ghost legs already included
                    shift = max(gadget_1.a, gadget_1.b)
                    pre_image = temp_reverse_map_2[(elem[0], 1)][0]
                    i2_shift_map[(pre_image + shift, 1)] = (elem[0], i2_index)
                    # print("TOOK")
                    # print((pre_image + shift, 1))
                    # print((elem[0], i2_index))
                    # if not ((pre_image + shift) in all_pre_images):
                    #     # Correction
                    #     print("NEED CORRECTION g0")
                    #     pass
                elif elem[0] in (g1_out): # note ghost legs already included
                    shift = max(gadget_0.a, gadget_0.b)
                    pre_image = temp_reverse_map_2[(elem[0], 1)][0]
                    # i2_shift_map[elem] = (new_i2_map[elem][0] - shift, new_i2_map[elem][1])
                    i2_shift_map[(pre_image - shift, 1)] = (elem[0], i2_index)
                    # print("TOOK")
                    # print((pre_image - shift, 1))
                    # print((elem[0], i2_index))
                    # if not ((pre_image - shift) in all_pre_images):
                    #     # Correction
                    #     print("NEED CORRECTION g1")
                    #     pass
                else:
                    # print("LEFT OVER")
                    # print(elem)
                    pre_image = temp_reverse_map_2[(elem[0], 1)][0]
                    if not(pre_image in all_pre_images_2):
                        i2_shift_map[(pre_image, 1)] = new_i2_map[(pre_image, 1)]
                    else:
                        continue
                    # i2_shift_map[elem] = new_i2_map[elem]

        # print("LEGS OUT OF LAST INTERLINK")
        # print(g0_out)
        # print(g1_out)
        # print(all_pre_images_2)

        # print("LINK 2 BEFORE SHIFTING")
        # for elem in new_i2_map.keys():
        #     print(elem, end=" to ")
        #     print(new_i2_map[elem])

        # print("LINK 2 AFTER SHIFTING")
        # for elem in i2_shift_map.keys():
        #     print(elem, end=" to ")
        #     print(i2_shift_map[elem])


        """
        *
        *
        * Step 3: adjusting the center interlink to reflect changes.
        *
        *
        """

        # Note: compute legs through the interlink for g0
        g0_legs = list(set([self.reverse_interlink(self.interlinks[i1_index])[(elem, 1)][0] for elem in (is_valid_0[2] + ghost_legs_f_2)]))
        g1_legs = list(set(is_valid_1[1] + ghost_legs_f_1))

        middle_interlink_map = self.interlinks[i1_index].map_to_grid
        for elem in middle_interlink_map.keys():
            if elem[1] == 1:
                if elem[0] in g0_legs:
                    new_i1_map[elem] = (elem[0], i1_index)
                    if not(middle_interlink.map_to_grid[(elem[0], 1)] in (g0_legs + g1_legs)):
                        image = middle_interlink.map_to_grid[(elem[0], 1)][0]
                        pre_image = self.reverse_interlink(middle_interlink)[(elem[0], 1)][0]
                        new_i1_map[(pre_image, 1)] = (image, i1_index)
                elif elem[0] in g1_legs:
                    new_i1_map[elem] = (elem[0], i1_index)
                    if not(middle_interlink.map_to_grid[(elem[0], 1)] in (g0_legs + g1_legs)):
                        image = middle_interlink.map_to_grid[(elem[0], 1)][0]
                        pre_image = self.reverse_interlink(middle_interlink)[(elem[0], 1)][0]
                        new_i1_map[(pre_image, 1)] = (image, i1_index)
                else:
                    if not (elem in new_i1_map.keys()):
                        new_i1_map[elem] = middle_interlink_map[elem]
            else:
                new_i1_map[elem] = middle_interlink_map[elem]

        """
        So you look at the legs you're set to change, take the complement, and then look forward through the original middle interlink. to where things map. Then look back through original first interlink, and see where it now maps. Then look through last interlink to see where it maps. Then see where these have been newly mapped to by shifting; and then connect up the remaining wires.
        """

        # We know that 2 and 3 map to 0 and 1
        # So we need to figure out the pre-image of 2 and 3 according to the new i0
        # pre-images are 2 and 3; after changing link 0 these go to 5 and 6
        # the after images of 0, 1 are, after shifting link 2, are 5, 6 as expected. So we want 5, 6 for the top of the interlink

        # Reversed shift maps
        temp_shift_i2 = Interlink(self.interlinks[i2_index].a, self.interlinks[i2_index].label, i2_shift_map)
        temp_shift_i2_map = self.reverse_interlink(temp_shift_i2)

        fixed_legs_0 = g0_legs
        fixed_legs_1 = [self.reverse_interlink(middle_interlink)[(elem, 1)][0] for elem in g1_legs]
        # print(fixed_legs_0 + fixed_legs_1)

        for elem in range(len(self.global_grid)):
            if not (elem in (fixed_legs_0 + fixed_legs_1)):
                forward_image = middle_interlink.map_to_grid[(elem, 1)][0]
                pre_image = self.reverse_interlink(first_interlink)[(elem, 1)][0]
                next_image = last_interlink.map_to_grid[(forward_image, 1)][0]
                now_maps_0 = i0_shift_map[(pre_image, 1)][0]
                now_maps_2 = temp_shift_i2_map[(next_image, 1)][0]
                # print(now_maps_0)
                # print(now_maps_2)
                new_i1_map[(now_maps_0, 1)] = (now_maps_2, i1_index)

        # print("MIDDLE LEGS")
        # print(g0_legs)
        # print(g1_legs)

        # print("MIDDLE INTERLINK BEFORE")
        # for elem in middle_interlink.map_to_grid.keys():
        #     print(elem, end=" to ")
        #     print(middle_interlink.map_to_grid[elem])

        # print("MIDDLE INTERLINK AFTER")
        # for elem in new_i1_map.keys():
        #     print(elem, end=" to ")
        #     print(new_i1_map[elem])

        # print("GADGET MAPS 0")
        # for elem in new_g0_map.keys():
        #     print(elem, end=" to ")
        #     print(new_g0_map[elem])

        # print("GADGET MAPS 1")
        # for elem in new_g1_map.keys():
        #     print(elem, end=" to ")
        #     print(new_g1_map[elem])

        """
        *
        *
        * Fourth step: instantiating new gadgets and interlinks
        *
        *
        """

        # Index gadgets
        g_index = comp_ends_0[0]

        new_g0 = Gadget(gadget_0.a, gadget_0.b, gadget_0.label, new_g0_map)
        new_g1 = Gadget(gadget_1.a, gadget_1.b, gadget_1.label, new_g1_map)

        new_i0 = Interlink(self.interlinks[i0_index].a, self.interlinks[i0_index].label, i0_shift_map)
        new_i1 = Interlink(self.interlinks[i1_index].a, self.interlinks[i1_index].label, new_i1_map)
        new_i2 = Interlink(self.interlinks[i2_index].a, self.interlinks[i2_index].label, i2_shift_map)

        new_gadgets = self.gadgets[0:g_index] + [new_g1, new_g0] + self.gadgets[g_index + 2:]
        new_interlinks = self.interlinks[0:i0_index] + [new_i0, new_i1, new_i2] + self.interlinks[i0_index + 3:]

        # for elem in new_gadgets:
        #     print(elem.label)

        # for elem in new_interlinks:
        #     print(elem.label)


        """
        *
        *
        * Fifth step: straigtening ghost legs to maintain desired property.
        *
        * NOTE: currently we have a limited set of these; we seem to be able to ignore re-indexing because the middle interlink becomes trivial for input and output legs, but it would be good to verify this with more checks in the future!
        """

        # Correct the right-leading ghost legs: note same index, as between the gadgets we have a flat expression
        for k in range(i2_index,len(new_interlinks)):
            # r0
            for elem in ghost_legs_f_2:
                shift_index = elem + max(gadget_1.a, gadget_1.b)
                maps_to = new_interlinks[k].map_to_grid[(shift_index, 1)][0]
                if maps_to != shift_index:
                    pre_image = self.reverse_interlink(new_interlinks[k])[(shift_index, 1)][0]
                    new_interlinks[k].map_to_grid[(shift_index, 1)] = (shift_index, k)
                    new_interlinks[k].map_to_grid[(pre_image, 1)] = (maps_to, k)
                else:
                    continue
            # r1
            for elem in ghost_legs_f_3:
                shift_index = elem - max(gadget_0.a, gadget_0.b)
                maps_to = new_interlinks[k].map_to_grid[(shift_index, 1)][0]
                if maps_to != shift_index:
                    pre_image = self.reverse_interlink(new_interlinks[k])[(shift_index, 1)][0]
                    new_interlinks[k].map_to_grid[(shift_index, 1)] = (shift_index, k)
                    new_interlinks[k].map_to_grid[(pre_image, 1)] = (maps_to, k)
                else:
                    continue

        for k in range(0, i0_index + 1):
            rev_k = i0_index - k
            # l0
            # print("GHOST L0")
            # print(ghost_legs_f_0)
            for elem in ghost_legs_f_0:
                shift_index = elem + max(gadget_1.a, gadget_1.b)
                maps_to = self.reverse_interlink(new_interlinks[rev_k])[(shift_index, 1)][0] # NOTE REVERSED
                if maps_to != shift_index:
                    image = new_interlinks[rev_k].map_to_grid[(shift_index, 1)][0]
                    new_interlinks[rev_k].map_to_grid[(shift_index, 1)] = (shift_index, k)
                    new_interlinks[rev_k].map_to_grid[(maps_to, 1)] = (image, k)
                else:
                    continue
            # l1
            for elem in ghost_legs_f_1:
                shift_index = elem - max(gadget_0.a, gadget_0.b)
                maps_to = self.reverse_interlink(new_interlinks[rev_k])[(shift_index, 1)][0] # NOTE REVERSED
                if maps_to != shift_index:
                    image = new_interlinks[rev_k].map_to_grid[(shift_index, 1)][0]
                    new_interlinks[rev_k].map_to_grid[(shift_index, 1)] = (shift_index, k)
                    new_interlinks[rev_k].map_to_grid[(maps_to, 1)] = (image, k)
                else:
                    continue

        # print("NEW INTERLINK GHOST 0")
        # for elem in new_interlinks[0].map_to_grid.keys():
        #     print(elem, end=" to ")
        #     print(new_interlinks[0].map_to_grid[elem])

        # print("NEW INTERLINK GHOST 1")
        # for elem in new_interlinks[1].map_to_grid.keys():
        #     print(elem, end=" to ")
        #     print(new_interlinks[1].map_to_grid[elem])

        # print("NEW INTERLINK GHOST 2")
        # for elem in new_interlinks[2].map_to_grid.keys():
        #     print(elem, end=" to ")
        #     print(new_interlinks[2].map_to_grid[elem])

        # print("NEW INTERLINK GHOST 3")
        # for elem in new_interlinks[3].map_to_grid.keys():
        #     print(elem, end=" to ")
        #     print(new_interlinks[3].map_to_grid[elem])

        """
        *
        *
        * Sixth step: instantiating new assemblage and returning
        *
        *
        """

        new_assemblage = GadgetAssemblage(new_gadgets, new_interlinks)

        return new_assemblage

    """
    Function which examines a selection of gadgets, and determines if they can all be swapped to be adjacent, and thus replaced by another gadget of a definite size. This involves trying to successively swap any non-consecutive gadgets through the collection. If possible, returns the swapped gadget, as well as a dictionary of its output legs.
    """
    def is_replaceable(self, gadgets):
        pass
    
    """
    This method works by sucessively swapping non-connected gadgets until a desired component is agglomerated.
    """
    def agglomerate_component(self, gadgets):
        pass
        """
        Given a valid component, and a map of the terminal legs, produces a new gadget and corresponding dict with the same form; this should be done in two steps. The only case is the three gadget one; we can do all of these updates and then refresh the dict (gadgets and wires) and check validity. Everything happens within the map_to_grid of gadgets and interlinks. Gadgets shift uniformly (along with coords) and the interlinks are made to follow.
        """

    def contract(self, gadgets):
        pass

    def expand(self, gadgets, gadget_assemblage):
        pass

    """
    Used to allow shorthand notation for linear specification of gadgets; not so obvious this can be done super cleanly, but you'd like people to be able to talk about local leg to local leg connections, and then the code above can check things.

    This actually shouldn't be a method of the assemblage class, just something that tries to build the right instantiation objects.
    """
    def build_maps(self, gadgets, interlinks):
        pass

    """
    Function to give simple list of tuples indicating the the local index and name of the input and output legs of assemblage; everything is indexed by the first leg it encounters.
    """
    def get_leg_guide(self):
        legs = self.get_terminal_legs()
        input_legs = list(map(lambda x: (x[0], x[1]), legs[2]))
        output_legs = list(map(lambda x: (x[0], x[1]), legs[3]))
        return (input_legs, output_legs)

    """
    Checks if the 
    """
    def is_valid_linkage(self, o_legs, i_legs, linkage):
        potential_o_legs = list(map(lambda x: x[0], linkage))
        potential_i_legs = list(map(lambda x: x[1], linkage))

        # Indicate the valid choices of input and output; and remove pairs every time one is encountered; if this ever fails to work, then throw an error.
        legs_remaining = set(list(o_legs.keys()) + list(i_legs.keys()))

        for k in range(len(linkage)):
            key_0 = (potential_o_legs[k][0], potential_o_legs[k][1], 1)
            key_1 = (potential_i_legs[k][0], potential_i_legs[k][1], 0)

            if key_0 in o_legs:
                if key_1 in i_legs:
                    if (key_0 in legs_remaining) and (key_1 in legs_remaining):
                        legs_remaining.remove(key_0)
                        legs_remaining.remove(key_1)
                    else:
                        raise NameError("Either leg %s or leg %s has been used already." % (str(key_0[0:2]), str(key_1[0:2])))
                else:
                    raise NameError("Leg %s not input for second assemblage." % str((potential_i_legs[k][0], potential_i_legs[k][1])))
            else:
                raise NameError("Leg %s not output by first assemblage." % str((potential_o_legs[k][0], potential_o_legs[k][1])))
        return True

    """
    This method should allow us to link things nicely, following an understanding of which legs can be linked; checks the terminal legs of each, checks for ghost legs, and then simply assembles to gadgets. Also need to fill in missing blocks elsewhere for all the relevant maps.
    """
    def link_assemblage(self, assemblage, linkage):
        legs_0 = self.get_terminal_legs()
        legs_1 = assemblage.get_terminal_legs()

        """
        Eventually we need to check that the linkage is valid; we need that the set of keys for the two dictionaries below are exactly those elements in the linkage object.
        """

        # Format of terminal legs(sorted(input_legs), sorted(output_legs), input_leg_dict, output_leg_dict)
        # Legs are labelled by gadget name and local leg.

        # linkage is a series of tuples with form ((g_name, leg), (g_name, leg))
        
        output_legs_0 = legs_0[3]
        input_legs_1 = legs_1[2]

        # print("OUTPUT LEGS of A0")
        # for elem in output_legs_0:
        #     print(elem)
        #     print(output_legs_0[elem])

        # Note that these will be uniformly shifted up; but when we are referencing them, they stay as they are.
        # print("INPUT LEGS of A1")
        # for elem in input_legs_1:
        #     print(elem)
        #     print(input_legs_1[elem])

        """
        SOMETHING TO ADD: we need to determine if a valid pairing has been specified; this means that the set of inputs and outputs have to be distinct, belong to the right sets, and have no duplicates. Beyond this, everything is permitted.
        """
        is_valid_linkage = self.is_valid_linkage(output_legs_0, input_legs_1, linkage)

        # Another thing to note is that the swaps on the central interlink wont cause any problems here; each necessarily is not a ghost leg, and can't be sent somewhere that a ghost leg exists.

        # Create a list of (a, b) tuples indicating that leg a on the first assemblage should be linked to leg b in the second. Note these are global coordinates, and are taken after the interlink.
        linkage_guide = []
        for elem in linkage:
            key_0 = (elem[0][0], elem[0][1], 1)
            key_1 = (elem[1][0], elem[1][1], 0)
            a0_leg = output_legs_0[key_0]
            a1_leg = input_legs_1[key_1] + len(self.global_grid)
            linkage_guide.append((a0_leg, a1_leg))

        # print(linkage_guide)

        new_gadgets = []
        new_interlinks = []

        # Add modified gadgets for first assemblage
        for g in self.gadgets:
            new_g_a = g.a
            new_g_b = g.b
            new_g_label = g.label
            new_g_map = copy.deepcopy(g.map_to_grid)
            # Add these unmodified values to new gadget list
            new_g = Gadget(new_g_a, new_g_b, new_g_label, new_g_map)
            new_gadgets.append(new_g)

        # Add modified gadgets for second assemblage
        for g in assemblage.gadgets:
            new_g_a = g.a
            new_g_b = g.b
            new_g_label = g.label
            old_g_map = copy.deepcopy(g.map_to_grid)
            new_g_map = dict()
            for elem in old_g_map.keys():
                old_elem = old_g_map[elem]
                y_shift = len(self.global_grid)
                x_shift = len(self.global_grid[0]) - 1 # Note that the gadgets shift by one less than the length
                new_elem = (old_elem[0] + y_shift, old_elem[1] + x_shift)
                new_g_map[elem] = new_elem
            # Add these unmodified values to new gadget list
            new_g = Gadget(new_g_a, new_g_b, new_g_label, new_g_map)
            new_gadgets.append(new_g)

        # Add modified interlinks for the first assemblage
        i_y_start = len(self.global_grid) # Find y offset for change.
        for i in self.interlinks:
            new_i_a = i.a + len(assemblage.global_grid)
            new_i_label = i.label 
            new_i_map = copy.deepcopy(i.map_to_grid) # NOTE COPY USE: PREVIOUSLY A MYSTERIOUS BUG FROM THIS!
            i_x_coord = i.map_to_grid[(0, 1)][1] # Find x offset for change.
            for k in range(len(assemblage.global_grid)):
                # We add both x = 0, 1 elements to the dictionary.
                new_i_map[(i_y_start + k, 1)] = (i_y_start + k, i_x_coord)
                new_i_map[(i_y_start + k, 0)] = (i_y_start + k, i_x_coord)
            new_i = Interlink(new_i_a, new_i_label, new_i_map)
            new_interlinks.append(new_i)

        # Add modified interlinks for the second assemblage
        for i in assemblage.interlinks:
            new_i_a = i.a + len(self.global_grid)
            new_i_label = i.label 
            old_i_map = copy.deepcopy(i.map_to_grid)
            old_i_x_coord = old_i_map[(0, 1)]
            new_i_map = dict()

            y_shift = len(self.global_grid)
            x_shift = len(self.global_grid[0]) # Note no shift

            # We want to ignore the first interlink of the second gadget
            for elem in old_i_map.keys():
                old_elem = old_i_map[elem]
                new_elem = (elem[0] + y_shift, elem[1])
                i_x_coord = old_elem[1]
                if i_x_coord != 0:
                    # Shift local y coord up, and shift global y and x coords up and out.
                    new_i_map[new_elem] = (old_elem[0] + y_shift, i_x_coord + x_shift - 1)
                else:
                    # Ignore the first interlink for now.
                    continue
                # Populate the missing bottom of each new interlink
                for k in range(len(self.global_grid)):
                    if i_x_coord != 0:
                        new_i_map[(k, 1)] = (k, i_x_coord + x_shift - 1) # Trivial interlink elements
                        new_i_map[(k, 0)] = (k, i_x_coord + x_shift - 1) # Trivial interlink elements
                    else:
                        continue
            if list(new_i_map.keys()) != []:
                new_i = Interlink(new_i_a, new_i_label, new_i_map)
                new_interlinks.append(new_i)
            else:
                continue

        middle_index = len(self.global_grid[0]) - 1
        middle_interlink = new_interlinks[middle_index]
        for elem in linkage_guide:

            maps_from = self.reverse_interlink(middle_interlink)[(elem[0], 1)][0]

            middle_interlink.map_to_grid[(maps_from, 1)] = (elem[1], middle_index)
            middle_interlink.map_to_grid[(elem[1], 1)] = (elem[0], middle_index)

            # middle_interlink.map_to_grid[(elem[0], 1)] = (elem[1], middle_index)
            # middle_interlink.map_to_grid[(elem[1], 1)] = (elem[0], middle_index)

        new_assemblage = GadgetAssemblage(new_gadgets, new_interlinks)
        return new_assemblage

    """
    Returns a special dictionary that indicates, for every position on the global grid, the first object (and local leg position) that that wire runs in to going left, in the form (g, local_y) for running into gadgets and ('WIRE', global_y) for running into an input leg.
    """
    def gen_leg_origin_guide(self):
        # Specify an empty dictionary which is dynamically updated.
        leg_origin_guide = dict()

        for x in range(len(self.global_grid[0])):
            for y in range(len(self.global_grid)):
                if (y, x) in leg_origin_guide:
                    continue
                else:
                    loc = self.gadget_dict[(y, x)]
                    has_terminated = False 
                    while not has_terminated:
                        if loc == (None, None):
                            has_terminated = True
                        else:
                            if loc[0][0] != "WIRE":
                                # Terminate.
                                leg_origin_guide[(y, x)] = (loc[0][0], loc[0][2][0]) # (g, y)
                                has_terminated = True
                            elif loc[0][2] == "TERMINAL":
                                leg_origin_guide[(y, x)] = (loc[0][0], loc[0][1][0]) # ('WIRE', y)
                                # Terminate.
                                has_terminated = True
                            else:
                                prev_loc = loc[0][1]
                                if prev_loc in leg_origin_guide:
                                    leg_origin_guide[(y, x)] = leg_origin_guide[prev_loc]
                                    has_terminated = True
                                else:
                                    loc = self.gadget_dict[prev_loc]
                                # Continue on.

        return leg_origin_guide

"""
An external function which takes a list of gadgets and returns a gadget assemblage in which they are implemented in parallel (i.e., without linking).
"""
def wrap_parallel_gadgets(gadgets):
    assemblage_list = []
    for g in gadgets:
        assemblage_list.append(g.wrap_gadget())
    current_assemblage = assemblage_list[0]
    
    # Successively trivially link assemblages together.
    for k in range(1, len(assemblage_list)):
        current_assemblage = current_assemblage.link_assemblage(assemblage_list[k], [])
    return current_assemblage

def main():
    pass

if __name__ == "__main__":
    main()