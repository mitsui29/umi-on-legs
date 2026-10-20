"""Prepare a portable URDF for Isaac Sim without changing the upstream asset."""

from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "resources/robots/go2_arx5/go2_arx5_finray_x85_z94.urdf"


def prepare_urdf(cache_dir):
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(URDF)
    for mesh in tree.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        # Upstream paths start at mani-centric-wbc, rather than the URDF folder.
        path = Path(filename)
        if not path.is_absolute():
            path = ROOT / filename.removeprefix("package://")
        if not path.is_file():
            raise FileNotFoundError(f"URDF mesh does not exist: {path}")
        mesh.set("filename", str(path.resolve()))
    target = cache / "go2_arx5.urdf"
    contents = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    if not target.exists() or target.read_bytes() != contents:
        target.write_bytes(contents)
    return str(target)
