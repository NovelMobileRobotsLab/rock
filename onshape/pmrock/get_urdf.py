import onshape_robotics_toolkit as osa
from onshape_robotics_toolkit.connect import Client
from onshape_robotics_toolkit.models.document import Document
from onshape_robotics_toolkit.log import LOGGER

import time

# get directory of this file
import os
onshape_dir = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != onshape_dir:
    print(f"Current directory: {os.getcwd()}")
    print(f"Please run this file from {onshape_dir}")
    print(f"Run the following in terminal:")
    print(f"cd {onshape_dir}")
    exit()

# Initialize the client
LOGGER._log_path = "logs"

#make a directory for logs if it doesn't exist
if not os.path.exists(LOGGER._log_path):
    os.makedirs(LOGGER._log_path)
client = osa.Client(
    env="./../.env"
)

doc = Document.from_url(
    url="https://cad.onshape.com/documents/e4e428d4cfd3a406f9a849f8/w/165e68061700713e3a2b8fda/e/be5e169679c58870424516dc"
)

# Retrieve the Variable Studio element
elements = client.get_elements(doc.did, doc.wtype, doc.wid)

print(f"Elements:\n {elements}")

# for i in range(22, 50, 2):
#     variables = client.get_variables(doc.did, doc.wid, elements["Variable Studio 1"].id)
#     variables["pendulum_radius"].expression = f"{i} mm"
#     variables["pendulum_height"].expression = f"{i} mm"
#     variables["shell_com_height"].expression = f"{i-12} mm"
#     client.set_variables(doc.did, doc.wid, elements["Variable Studio 1"].id, variables)
#     print(i)
#     time.sleep(0.1)
# exit()
# variables["wheelThickness"].expression = "71 mm"
# variables["forkAngle"].expression = "20 deg"

# Save the updated variables back to the Variable Studio
variables = client.get_variables(doc.did, doc.wid, elements["Variable Studio 1"].id)
variables["pendulum_height"].expression = f"{300} mm"
variables["shell_com_height"].expression = f"{288} mm"
client.set_variables(doc.did, doc.wid, elements["Variable Studio 1"].id, variables)

from onshape_robotics_toolkit.parse import (
    get_instances,
    get_mates_and_relations,
    get_occurrences,
    get_parts,
    get_subassemblies,
)

# Retrieve the assembly
assembly = client.get_assembly(doc.did, doc.wtype, doc.wid, elements["Assembly 1"].id)

# Extract components
instances, occurrences, id_to_name_map = get_instances(assembly, max_depth=1)

subassemblies, rigid_subassemblies = get_subassemblies(assembly, client, instances)
parts = get_parts(assembly, rigid_subassemblies, client, instances)

print(parts)

with open('parts.json', 'w') as f:
    print(parts, file=f)


mates, relations = get_mates_and_relations(assembly, subassemblies, rigid_subassemblies, id_to_name_map, parts)

from onshape_robotics_toolkit.graph import create_graph
from onshape_robotics_toolkit.robot import get_robot
from onshape_robotics_toolkit.robot import Robot

# Create and save the assembly graph
graph, root_node = create_graph(occurrences=occurrences, instances=instances, parts=parts, mates=mates)

robot = get_robot(assembly, graph, root_node, parts, mates, relations, client, "test")
robot.show_graph("rock1.png")

mjcf_str = robot.to_urdf()

# with open("rock1.mjcf", "w", encoding="utf-8") as f:
#     f.write(mjcf_str)

robot.save("rock1.urdf")
