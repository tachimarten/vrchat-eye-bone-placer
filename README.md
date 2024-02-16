# VRChat Eye Bone Placer

VRChat assumes that eyeballs are spheres. But sometimes they aren't.
This is where this tool comes in.

To have a non-spherical eye in VRChat, make the eyeball a static sphere, and attach the iris to a bone. This bone needs to be placed in such a way that it glides
over the eyeball as close as possible without going inside. Placing this is a pain, so this tool automates it.

Choose a mesh in `.obj` format containing the **visible part** of **one**
eyeball, and this tool will tell you where to place the bone.  This only places
one bone, so make sure to only supply a single eyeball. You may mirror the bone
in your modeling software to produce the other one. Also, it helps if you delete
parts of the mesh that aren't visible, so that they aren't included in the
approximation. Finally, remember that Blender switches the Y and Z axes whenever
exporting in `.obj` format, so you may have to switch them back when entering
the result in Blender.

Note that it may take a few seconds to place the bone when you click "Place Eye Bone". This is normal.
