from syft.lib.python.int import Int

int_1 = Int(value=1)
int_2 = Int(value=2)

int_add_1 = int_1 + int_2
int_add_2 = 1 + int_2
int_add_3 = int_2 + 1
assert int_add_1 == int_add_2 == int_add_3 == 3
assert isinstance(int_1, int)

int_sub_1 = int_1 - int_2
int_sub_2 = 1 - int_2
int_sub_3 = int_1 - 2
assert int_sub_1 == int_sub_2 == int_sub_3
