import xml.etree.ElementTree as ET


# Parse the XDMF file
tree = ET.parse('displacement.xdmf')
trees = tree.findall(".//{*}include")
root = tree.getroot()

grids = tree.findall(".//Grid")
mesh_tag_list = []
for i, root_child in enumerate(grids):
    if "Name" in root_child.keys() and "GridType" in root_child.keys() and "CollectionType" not in root_child.keys():
        mesh_tag_list.append(i)

fix_start_index = mesh_tag_list[-1] - 2


xpointer_string = f"xpointer(//Grid[@Name=&quot;TimeSeries_Velocity&quot;]/Grid[{mesh_tag_list[-1]}]/*[self::Topology or self::Geometry])"
for i in range(fix_start_index, len(trees)):
    wrong_tree = trees[i]
    wrong_tree.clear()
    wrong_tree.set("xpointer", xpointer_string)


# Save the modified XML back to a file
tree.write('fixed_displacement.xdmf', encoding='utf-8', xml_declaration=True)
