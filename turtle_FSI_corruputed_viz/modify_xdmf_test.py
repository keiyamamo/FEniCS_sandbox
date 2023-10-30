import xml.etree.ElementTree as ET
# Define the namespace used in the XML
namespace = {'ns0': 'http://www.w3.org/2001/XInclude'}

# Parse the XDMF file
tree = ET.parse('velocity.xdmf')
root = tree.getroot()

# Find the specific <ns0:include> element you want to modify
target_include = root.find(".//<ns0:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_Velocity&quot;]/Grid[1]/*[self::Topology or self::Geometry], namespaces=namespace)

# Check if the target <ns0:include> element is found
if target_include is not None:
    # Modify the xpointer attribute to reference Grid[4]
    target_include.set("xpointer", 'xpointer(//Grid[@Name="TimeSeries_Velocity"]/Grid[4]/*[self::Topology or self::Geometry])')

# Save the modified XML back to a file
tree.write('modified_xdmf_file.xml')
