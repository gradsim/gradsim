import kaolin
from pathlib import Path


class TriangleMesh:

    def from_obj(obj_path: Path):
        mesh = kaolin.io.obj.import_mesh(str(obj_path))
        return mesh
    
