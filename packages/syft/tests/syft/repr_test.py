# stdlib

# third party

# syft absolute

# def test_list_repr_html():
#     obj = SyftObject()
#     list_obj = [obj]
#     string_values = pd.read_html(list_obj._repr_html_()[15:])[0].values[0]
#     type_name = string_values[1]
#     id_repr = string_values[2]
#     assert "." not in type_name
#     assert id_repr[:6] == re.search("[0-9a-f]", id_repr[:6]).string
#     assert id_repr[6:] == "..."


# def test_dict_repr_html():
#     obj = SyftObject()
#     dict_obj = {"key": obj}
#     string_values = pd.read_html(dict_obj._repr_html_()[15:])[0].values[0]
#     type_name = string_values[2]
#     id_repr = string_values[3]
#     assert "." not in type_name
#     assert id_repr[:6] == re.search("[0-9a-f]", id_repr[:6]).string
#     assert id_repr[6:] == "..."
