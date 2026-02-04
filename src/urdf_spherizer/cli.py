from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from urdf_spherizer._core import spherize_mesh


def parse_vector(text: Optional[str], count: int, default: Tuple[float, ...]) -> Tuple[float, ...]:
    if text is None:
        return default
    parts = text.strip().split()
    if len(parts) != count:
        raise ValueError(f"Expected {count} values, got {len(parts)}: {text!r}")
    return tuple(float(p) for p in parts)


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def build_transform(xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> np.ndarray:
    rot = rpy_to_matrix(*rpy)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rot
    transform[:3, 3] = np.array(xyz, dtype=np.float32)
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return points @ rot.T + trans


def resolve_mesh_path(filename: str, urdf_dir: Path, package_map: Dict[str, Path]) -> Path:
    if filename.startswith("package://"):
        suffix = filename[len("package://") :]
        if "/" not in suffix:
            raise ValueError(f"package:// URL missing path: {filename}")
        package, rel = suffix.split("/", 1)
        if package not in package_map:
            raise ValueError(
                f"Package {package!r} not found. Use --package-dir {package}=PATH."
            )
        return package_map[package] / rel
    path = Path(filename)
    if path.is_absolute():
        return path
    return urdf_dir / path


def load_trimesh(path: Path) -> Optional[trimesh.Trimesh]:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            return None
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    if mesh.is_empty:
        return None
    return mesh


def build_visual_mesh(
    visual_elem: ET.Element, urdf_dir: Path, package_map: Dict[str, Path]
) -> Optional[trimesh.Trimesh]:
    geometry = visual_elem.find("geometry")
    if geometry is None:
        return None

    mesh: Optional[trimesh.Trimesh] = None
    mesh_elem = geometry.find("mesh")
    if mesh_elem is not None and "filename" in mesh_elem.attrib:
        filename = mesh_elem.attrib["filename"]
        mesh_path = resolve_mesh_path(filename, urdf_dir, package_map)
        mesh = load_trimesh(mesh_path)
        if mesh is None:
            return None
        scale = parse_vector(mesh_elem.attrib.get("scale"), 3, (1.0, 1.0, 1.0))
        mesh = mesh.copy()
        mesh.apply_scale(np.array(scale, dtype=np.float32))
    else:
        box_elem = geometry.find("box")
        cylinder_elem = geometry.find("cylinder")
        sphere_elem = geometry.find("sphere")
        if box_elem is not None and "size" in box_elem.attrib:
            size = parse_vector(box_elem.attrib.get("size"), 3, (1.0, 1.0, 1.0))
            mesh = trimesh.creation.box(extents=np.array(size, dtype=np.float32))
        elif cylinder_elem is not None:
            radius = float(cylinder_elem.attrib.get("radius", "0"))
            length = float(cylinder_elem.attrib.get("length", "0"))
            if radius > 0 and length > 0:
                mesh = trimesh.creation.cylinder(
                    radius=radius, height=length, sections=24
                )
        elif sphere_elem is not None:
            radius = float(sphere_elem.attrib.get("radius", "0"))
            if radius > 0:
                mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)

    if mesh is None:
        return None

    origin_elem = visual_elem.find("origin")
    xyz = parse_vector(origin_elem.attrib.get("xyz") if origin_elem is not None else None, 3, (0.0, 0.0, 0.0))
    rpy = parse_vector(origin_elem.attrib.get("rpy") if origin_elem is not None else None, 3, (0.0, 0.0, 0.0))
    transform = build_transform(xyz, rpy)
    mesh = mesh.copy()
    mesh.apply_transform(transform)
    return mesh


def build_link_mesh(
    link_elem: ET.Element, urdf_dir: Path, package_map: Dict[str, Path]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    meshes: List[trimesh.Trimesh] = []
    for visual in link_elem.findall("visual"):
        mesh = build_visual_mesh(visual, urdf_dir, package_map)
        if mesh is not None:
            meshes.append(mesh)
    if not meshes:
        return None
    if len(meshes) == 1:
        combined = meshes[0]
    else:
        combined = trimesh.util.concatenate(meshes)
    vertices = np.asarray(combined.vertices, dtype=np.float32)
    faces = np.asarray(combined.faces, dtype=np.uint32)
    if vertices.size == 0 or faces.size == 0:
        return None
    return vertices, faces


def format_vec(values: Iterable[float]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def replace_link_collisions(link_elem: ET.Element, spheres: List[Tuple[List[float], float]]) -> None:
    for collision in list(link_elem.findall("collision")):
        link_elem.remove(collision)
    for idx, (center, radius) in enumerate(spheres):
        collision = ET.SubElement(link_elem, "collision", {"name": f"sphere_{idx}"})
        ET.SubElement(collision, "origin", {"xyz": format_vec(center), "rpy": "0 0 0"})
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "sphere", {"radius": f"{radius:.6g}"})


def parse_package_dirs(values: List[str]) -> Dict[str, Path]:
    package_map: Dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --package-dir entry: {item!r} (expected NAME=PATH)")
        name, path = item.split("=", 1)
        package_map[name] = Path(path)
    return package_map


def compute_link_transforms(root_elem: ET.Element) -> Dict[str, np.ndarray]:
    parent_map: Dict[str, Tuple[str, np.ndarray]] = {}
    children_map: Dict[str, List[str]] = {}
    for joint in root_elem.findall("joint"):
        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parent = parent_elem.attrib.get("link")
        child = child_elem.attrib.get("link")
        if not parent or not child:
            continue
        origin_elem = joint.find("origin")
        xyz = parse_vector(
            origin_elem.attrib.get("xyz") if origin_elem is not None else None,
            3,
            (0.0, 0.0, 0.0),
        )
        rpy = parse_vector(
            origin_elem.attrib.get("rpy") if origin_elem is not None else None,
            3,
            (0.0, 0.0, 0.0),
        )
        transform = build_transform(xyz, rpy)
        parent_map[child] = (parent, transform)
        children_map.setdefault(parent, []).append(child)

    link_names = [link.attrib.get("name", "link") for link in root_elem.findall("link")]
    child_links = set(parent_map.keys())
    root_links = [name for name in link_names if name not in child_links]
    if not root_links:
        root_links = link_names

    transforms: Dict[str, np.ndarray] = {}

    def dfs(link_name: str, current: np.ndarray) -> None:
        if link_name in transforms:
            return
        transforms[link_name] = current
        for child in children_map.get(link_name, []):
            parent, local_tf = parent_map[child]
            if parent != link_name:
                continue
            dfs(child, current @ local_tf)

    identity = np.eye(4, dtype=np.float32)
    for root in root_links:
        dfs(root, identity)
    return transforms


def log_link_viz(
    rr_module, link_name: str, vertices: np.ndarray, faces: np.ndarray, spheres
) -> None:
    rr_module.log(
        f"link/{link_name}/visual",
        rr_module.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=faces,
        ),
    )
    if spheres:
        centers = np.asarray([s[0] for s in spheres], dtype=np.float32)
        radii = np.asarray([s[1] for s in spheres], dtype=np.float32)
        half_sizes = np.repeat(radii[:, None], 3, axis=1)
        rr_module.log(
            f"link/{link_name}/spheres",
            rr_module.Ellipsoids3D(
                centers=centers,
                half_sizes=half_sizes,
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sphere collisions from URDF visuals.")
    parser.add_argument("input", help="Input URDF path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output URDF path (default: <input>.spheres.urdf)",
    )
    parser.add_argument(
        "--max-spheres",
        type=int,
        default=64,
        help="Maximum spheres per link",
    )
    parser.add_argument(
        "--min-gain-ratio",
        type=float,
        default=0.02,
        help="Stop splitting when relative gain falls below this ratio (combined with --margin)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Stop splitting when absolute gain falls below this distance (meters)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-6,
        help="Radius inflation for numerical safety",
    )
    parser.add_argument(
        "--package-dir",
        action="append",
        default=[],
        help="Resolve package://NAME/... using NAME=PATH (can be repeated)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Visualize results with rerun",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input URDF not found: {input_path}")
    output_path = Path(args.output) if args.output else input_path.with_suffix(".spheres.urdf")
    package_map = parse_package_dirs(args.package_dir)

    tree = ET.parse(input_path)
    root = tree.getroot()
    urdf_dir = input_path.parent
    link_transforms = compute_link_transforms(root)

    rr_module = None
    if args.viz:
        import rerun as rr

        rr.init("urdf-spherizer", spawn=True)
        rr_module = rr

    for link in root.findall("link"):
        link_name = link.attrib.get("name", "link")
        mesh_data = build_link_mesh(link, urdf_dir, package_map)
        if mesh_data is None:
            continue
        vertices, faces = mesh_data
        spheres = spherize_mesh(
            vertices,
            faces,
            args.max_spheres,
            min_gain_ratio=args.min_gain_ratio,
            epsilon=args.epsilon,
            min_gain_abs=args.margin,
        )
        spheres = [(list(center), float(radius)) for center, radius in spheres]
        replace_link_collisions(link, spheres)
        if rr_module is not None:
            transform = link_transforms.get(link_name)
            if transform is not None:
                viz_vertices = apply_transform(vertices, transform)
                if spheres:
                    centers = np.asarray([s[0] for s in spheres], dtype=np.float32)
                    centers = apply_transform(centers, transform)
                    radii = [s[1] for s in spheres]
                    viz_spheres = [
                        (center.tolist(), radius) for center, radius in zip(centers, radii)
                    ]
                else:
                    viz_spheres = []
            else:
                viz_vertices = vertices
                viz_spheres = spheres
            log_link_viz(rr_module, link_name, viz_vertices, faces, viz_spheres)

    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
