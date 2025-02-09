import onshape_robotics_toolkit as osa
from onshape_robotics_toolkit.connect import Client
from onshape_robotics_toolkit.models.document import Document

# Initialize the client
client = osa.Client(
    env="./.env"
)

doc = Document.from_url(
    url="https://cad.onshape.com/documents/ca20d2f1622a5c39ca976405/w/3d8eb311791b3275fabcd834/e/41337b17eb94f64bb98f0912"
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
assembly = client.get_assembly(doc.did, doc.wtype, doc.wid, elements["assembly"].id)

# Extract components
instances, occurrences, id_to_name_map = get_instances(assembly, max_depth=1)

subassemblies, rigid_subassemblies = get_subassemblies(assembly, client, instances)
parts = get_parts(assembly, rigid_subassemblies, client, instances)

import json

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

with open("rock1.mjcf", "w", encoding="utf-8") as f:
    f.write(mjcf_str)

robot.save("rock1.urdf")
