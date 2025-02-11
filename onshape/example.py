import onshape_robotics_toolkit as osa
from onshape_robotics_toolkit.connect import Client
from onshape_robotics_toolkit.models.document import Document
from onshape_robotics_toolkit.log import LOGGER

# get directory of this file
import os
onshape_dir = os.path.dirname(os.path.abspath(__file__))

print(os.getcwd())
if os.getcwd() != onshape_dir:
    print(f"Current directory: {os.getcwd()}")
    print(f"Please run this file from {onshape_dir}")
    print(f"Run the following in terminal:")
    print(f"cd {onshape_dir}")
    exit()

# Initialize the client
LOGGER._log_path = f"{onshape_dir}/logs"
client = osa.Client(
    env=f"{onshape_dir}/.env"
)

doc = Document.from_url(
    url="https://cad.onshape.com/documents/e58f809a1903266b25fe8a9a/w/7677e945c3ad2c0522ac620f/e/b9b0680502e420b7cf8928f3"
)

# Retrieve the Variable Studio element
elements = client.get_elements(doc.did, doc.wtype, doc.wid)

print(f"Elements:\n {elements}")

# variables = client.get_variables(doc.did, doc.wid, elements["variables"].id)
# variables["wheelDiameter"].expression = "300 mm"
# variables["wheelThickness"].expression = "71 mm"
# variables["forkAngle"].expression = "20 deg"

# Save the updated variables back to the Variable Studio
# client.set_variables(doc.did, doc.wid, elements["variables"].id, variables)

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

import json

with open(f'{onshape_dir}/parts.json', 'w') as f:
    print(parts, file=f)


mates, relations = get_mates_and_relations(assembly, subassemblies, rigid_subassemblies, id_to_name_map, parts)

from onshape_robotics_toolkit.graph import create_graph
from onshape_robotics_toolkit.robot import get_robot
from onshape_robotics_toolkit.robot import Robot

# Create and save the assembly graph
graph, root_node = create_graph(occurrences=occurrences, instances=instances, parts=parts, mates=mates)

robot = get_robot(assembly, graph, root_node, parts, mates, relations, client, "test")
robot.show_graph(f"{onshape_dir}/rock1.png")

mjcf_str = robot.to_urdf()

# with open("rock1.mjcf", "w", encoding="utf-8") as f:
#     f.write(mjcf_str)

robot.save(f"{onshape_dir}/rock1.urdf")
