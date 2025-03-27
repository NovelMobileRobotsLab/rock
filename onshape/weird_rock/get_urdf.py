import onshape_robotics_toolkit as osa
from onshape_robotics_toolkit.connect import Client
from onshape_robotics_toolkit.models.document import Document
from onshape_robotics_toolkit.log import LOGGER
from onshape_robotics_toolkit.graph import create_graph
from onshape_robotics_toolkit.robot import get_robot
from onshape_robotics_toolkit.models.variable import Variable
from onshape_robotics_toolkit.parse import (
    get_instances,
    get_mates_and_relations,
    get_occurrences,
    get_parts,
    get_subassemblies,
)


# Ensure running from directory of this file
import os
onshape_dir = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != onshape_dir:
    print(f"Current directory: {os.getcwd()}")
    print(f"Please run this file from {onshape_dir}")
    print(f"Run the following in terminal:")
    print(f"cd {onshape_dir}")
    exit()

# CHANGE THESE FOR NEW FILES
save_name = "hellorock_elec"
url = "https://cad.onshape.com/documents/c330ce83f7b3527be6ffef2b/w/e336b9cb14c1d649c69b79d9/e/81d79c19a2e05dc4d8dbf4fd"
var_studio_name = "Variable Studio 1"
main_assembly_name = "hellorock"

# Set log directory
LOGGER._log_path = "logs"
if not os.path.exists(LOGGER._log_path):
    os.makedirs(LOGGER._log_path)

# Initialize the client with the access keys stored in the .env file
client = osa.Client(env="./../.env")

# Get the document from onshape URL
doc = Document.from_url(url)

# References to the Part Studios, Assemblies, and Variable Studios are stored in the elements dictionary
elements = client.get_elements(doc.did, doc.wtype, doc.wid)
print(f"Elements:\n {elements.keys()}")

# # Change the values of the variables if needed
# variables = client.get_variables(doc.did, doc.wid, elements[var_studio_name].id)
# print(f"Variables:\n {variables.keys()}")

# variables["x2"].expression = f"{-70} mm"
# variables['testvar'] = Variable(type="LENGTH", name="testvar", value="012 mm")

# for key in variables.keys():
#     if variables[key].expression == "":
#         variables[key].expression = f"{0} mm"
#     print(f"{key}: {variables[key]}")
# resp = client.set_variables(doc.did, doc.wid, elements[var_studio_name].id, variables)
# print(f"Response:\n {resp}")

# Retrieve the assembly you want to convert to URDF
assembly = client.get_assembly(doc.did, doc.wtype, doc.wid, elements[main_assembly_name].id)
instances, occurrences, id_to_name_map = get_instances(assembly, max_depth=1)
subassemblies, rigid_subassemblies = get_subassemblies(assembly, client, instances)
parts = get_parts(assembly, rigid_subassemblies, client, instances)
print(parts)
mates, relations = get_mates_and_relations(assembly, subassemblies, rigid_subassemblies, id_to_name_map, parts)

# Create and save the assembly graph
graph, root_node = create_graph(occurrences=occurrences, instances=instances, parts=parts, mates=mates)
robot = get_robot(assembly, graph, root_node, parts, mates, relations, client, save_name)
robot.show_graph(f"{save_name}.png")
robot.save(f"{save_name}.urdf")
