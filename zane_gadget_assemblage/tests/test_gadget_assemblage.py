import unittest
from gadgets import *

"""
Note these tests can be run with 'python -m unittest tests.test_gadget_assemblage' from outside the folder tests
"""

class TestGadgetAssemblageMethods(unittest.TestCase):

    def setUp(self):
        self.g0 = Gadget(1, 2, "g0")
        self.g1 = Gadget(1, 2, "g1")
        self.g2 = Gadget(1, 2, "g2")

        self.a0 = self.g0.wrap_gadget()
        self.a1 = self.g1.wrap_gadget()
        self.a2 = self.g2.wrap_gadget()

        self.a3 = self.a0.link_assemblage(self.a1, [(("g0", 0), ("g1", 0))])
        self.a4 = self.a3.link_assemblage(self.a2, [(("g1", 0), ("g2", 0))])

    def test_global_grid_size(self):
        y_size = len(self.a4.global_grid)
        x_size = len(self.a4.global_grid[0])
        self.assertEqual(y_size, 6)
        self.assertEqual(x_size, 4)

    def test_composite_gadget_size(self):
        a, b = self.a4.shape
        self.assertEqual(a, 1)
        self.assertEqual(b, 4)

    def test_input_output_legs(self):
        i_legs, o_legs, i_dict, o_dict = self.a4.get_terminal_legs()
        self.assertEqual(i_legs, [0])
        self.assertEqual(o_legs, [1, 3, 4, 5])
        self.assertEqual(i_dict[("g0", 0, 0)], 0)
        self.assertEqual(o_dict[("g2", 1, 1)], 5)

    def test_gadget_swap_size_invariant(self):
        h0 = Gadget(2, 2, "h0")
        h1 = Gadget(2, 2, "h1")

        b0 = h0.wrap_gadget()
        b1 = h1.wrap_gadget()

        b3 = b0.link_assemblage(b1, [])
        # Note call to the gadgets in the wrapped assemblage, not h0 and h1.
        b4 = b3.swap_adjacent_gadgets(b3.gadgets[0], b3.gadgets[1])

        in_0, out_0 = b3.shape
        in_1, out_1 = b4.shape

        self.assertEqual(in_0, in_1)
        self.assertEqual(out_0, out_1)

    def test_gadget_swap_terminal_legs_invariant(self):
        h0 = Gadget(2, 2, "h0")
        h1 = Gadget(2, 2, "h1")

        b0 = h0.wrap_gadget()
        b1 = h1.wrap_gadget()

        b3 = b0.link_assemblage(b1, [])
        # Note call to the gadgets in the wrapped assemblage, not h0 and h1.
        b4 = b3.swap_adjacent_gadgets(b3.gadgets[0], b3.gadgets[1])

        i_legs_0, o_legs_0, i_dict_0, o_dict_0 = b3.get_terminal_legs()
        i_legs_1, o_legs_1, i_dict_1, o_dict_1 = b4.get_terminal_legs()

        self.assertEqual(set(i_dict_0.keys()), set(i_dict_1.keys()))
        self.assertEqual(set(o_dict_0.keys()), set(o_dict_1.keys()))

    def test_gadget_name_swap_size_invariant(self):
        h0 = Gadget(2, 2, "h0")
        h1 = Gadget(2, 2, "h1")

        b0 = h0.wrap_gadget()
        b1 = h1.wrap_gadget()

        b3 = b0.link_assemblage(b1, [])
        # Note call to the gadgets by names h0 and h1.
        b4 = b3.swap_gadgets("h0", "h1")

        in_0, out_0 = b3.shape
        in_1, out_1 = b4.shape

        self.assertEqual(in_0, in_1)
        self.assertEqual(out_0, out_1)

    def test_bespoke_gadget_definition(self):
        # Define global grid dictionaries explicitly.
        map0 = {(0, 0):(0, 0), (0, 1):(0, 1), (1, 1):(1, 1)}
        map1 = {(0, 0):(2, 1), (0, 1):(2, 2), 
                (1, 0):(3, 1)}
        map2 = {(0, 0):(4, 2), (0, 1):(4, 3), 
                (1, 0):(5, 2), (1, 1):(5, 3), (2, 0):(6, 2)}
        map3 = {(0, 0):(0, 0), (1, 0):(1, 0), (2, 0):(2, 0), 
                (3, 0):(3, 0), (4, 0):(4, 0), (5, 0):(5, 0),(6, 0):(6, 0), 
                (0, 1):(5, 0), (1, 1):(1, 0), (2, 1):(2, 0), 
                (3, 1):(0, 0), (4, 1):(4, 0), (5, 1):(3, 0),(6, 1):(6, 0)}
        map4 = {(0, 0):(0, 1), (1, 0):(1, 1), (2, 0):(2, 1), 
                (3, 0):(3, 1), (4, 0):(4, 1), (5, 0):(5, 1),(6, 0):(6, 1),
                (0, 1):(0, 1), (1, 1):(3, 1), (2, 1):(2, 1), 
                (3, 1):(1, 1), (4, 1):(6, 1), (5, 1):(5, 1),(6, 1):(4, 1)}
        map5 = {(0, 0):(0, 2), (1, 0):(1, 2), (2, 0):(2, 2), 
                (3, 0):(3, 2), (4, 0):(4, 2), (5, 0):(5, 2),(6, 0):(6, 2),
                (0, 1):(0, 2), (1, 1):(2, 2), (2, 1):(1, 2), 
                (3, 1):(3, 2), (4, 1):(5, 2), (5, 1):(4, 2),(6, 1):(6, 2),}
        map6 = {(0, 0):(0, 3), (1, 0):(1, 3), (2, 0):(2, 3), 
                (3, 0):(3, 3), (4, 0):(4, 3), (5, 0):(5, 3),(6, 0):(6, 3),
                (0, 1):(4, 3), (1, 1):(0, 3), (2, 1):(1, 3), 
                (3, 1):(3, 3), (4, 1):(5, 3), (5, 1):(2, 3),(6, 1):(6, 3),}

        # Construct gadgets.
        g0 = Gadget(1, 2, "g0", map0)
        g1 = Gadget(2, 1, "g1", map1)
        g2 = Gadget(3, 2, "g2", map2)

        # Construct interlinks.
        i0 = Interlink(7, "i0", map3)
        i1 = Interlink(7, "i1", map4)
        i2 = Interlink(7, "i2", map5)
        i3 = Interlink(7, "i3", map6)

        # Assemble gadgets and interlinks.
        assemblage = GadgetAssemblage([g0, g1, g2], [i0, i1, i2, i3])
        # Swap assemblage gadgets g1 and g2, which are not causally related.
        new_assemblage = assemblage.swap_gadgets("g1", "g2")

        y_size_0 = len(assemblage.global_grid)
        x_size_0 = len(assemblage.global_grid[0])
        y_size_1 = len(new_assemblage.global_grid)
        x_size_1 = len(new_assemblage.global_grid[0])

        # Check size invariance.
        self.assertEqual(y_size_0, 7)
        self.assertEqual(x_size_0, 4)
        self.assertEqual(y_size_1, 7)
        self.assertEqual(x_size_1, 4)

        # Check reordering has occured.
        self.assertEqual(assemblage.gadgets[2].label, new_assemblage.gadgets[1].label)
        self.assertEqual(assemblage.gadgets[1].label, new_assemblage.gadgets[2].label)

        # Check that leg labels have been preserved in swap.
        i_legs_0, o_legs_0, i_dict_0, o_dict_0 = assemblage.get_terminal_legs()
        i_legs_1, o_legs_1, i_dict_1, o_dict_1 = new_assemblage.get_terminal_legs()
        self.assertEqual(set(i_dict_0.keys()), set(i_dict_1.keys()))
        self.assertEqual(set(o_dict_0.keys()), set(o_dict_1.keys()))

    def test_gadget_parallel_wrapping(self):
        g0 = Gadget(1, 2, "g0")
        g1 = Gadget(1, 2, "g1")
        g2 = Gadget(1, 2, "g2")
        g3 = Gadget(4, 5, "g3")
        g_list = [g0, g1, g2, g3]

        a0 = wrap_parallel_gadgets(g_list)

        # Check gadget size
        y_size = len(a0.global_grid)
        x_size = len(a0.global_grid[0])
        self.assertEqual(y_size, 11)
        self.assertEqual(x_size, 5)

        # Check input and output leg numbers.
        i_legs_0, o_legs_0, i_dict_0, o_dict_0 = a0.get_terminal_legs()
        in_leg_num = sum([g.a for g in g_list])
        out_leg_num = sum([g.b for g in g_list])
        self.assertEqual(len(i_legs_0), in_leg_num)
        self.assertEqual(len(o_legs_0), out_leg_num)

        # Generate two more parallel gadgets.
        h0 = Gadget(1, 2, "h0")
        h1 = Gadget(1, 2, "h1")
        h_list = [h0, h1]
        a1 = wrap_parallel_gadgets(h_list)

        # Trivially link assemblages.
        a2 = a0.link_assemblage(a1, [])

        # Check resulting size.
        y_size = len(a2.global_grid)
        x_size = len(a2.global_grid[0])
        self.assertEqual(y_size, 15)
        self.assertEqual(x_size, 7)

        # Check overall leg size again.
        i_legs_0, o_legs_0, i_dict_0, o_dict_0 = a2.get_terminal_legs()
        in_leg_num = sum([g.a for g in g_list] + [h.a for h in h_list])
        out_leg_num = sum([g.b for g in g_list] + [h.b for h in h_list])
        self.assertEqual(len(i_legs_0), in_leg_num)
        self.assertEqual(len(o_legs_0), out_leg_num)

if __name__ == '__main__':
    unittest.main()