use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    fn axis(self, axis: usize) -> f32 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }
}

impl From<[f32; 3]> for Vec3 {
    fn from(value: [f32; 3]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<Vec3> for [f32; 3] {
    fn from(value: Vec3) -> Self {
        [value.x, value.y, value.z]
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, scalar: f32) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Sphere {
    center: Vec3,
    radius: f32,
}

#[derive(Clone, Debug)]
struct Triangle {
    indices: [usize; 3],
    centroid: Vec3,
}

#[derive(Clone, Debug)]
struct Cluster {
    tris: Vec<usize>,
    sphere: Sphere,
}

impl PartialEq for Cluster {
    fn eq(&self, other: &Self) -> bool {
        self.sphere.radius == other.sphere.radius
    }
}

impl Eq for Cluster {}

impl PartialOrd for Cluster {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.sphere.radius.partial_cmp(&other.sphere.radius)
    }
}

impl Ord for Cluster {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone, Debug)]
struct SplitEval {
    left_tris: Vec<usize>,
    right_tris: Vec<usize>,
    left_sphere: Sphere,
    right_sphere: Sphere,
    gain_ratio: f32,
    gain_abs: f32,
    cost: f32,
}

fn ritter_sphere(points: &[Vec3], start: Vec3, epsilon: f32) -> Sphere {
    let mut p0 = start;
    let mut p1 = p0;
    let mut max_dist = 0.0_f32;
    for &p in points.iter() {
        let d = p0.distance(p);
        if d > max_dist {
            max_dist = d;
            p1 = p;
        }
    }
    let mut p2 = p1;
    max_dist = 0.0;
    for &p in points.iter() {
        let d = p1.distance(p);
        if d > max_dist {
            max_dist = d;
            p2 = p;
        }
    }
    let mut center = (p1 + p2) * 0.5;
    let mut radius = center.distance(p2);
    for &p in points.iter() {
        let d = center.distance(p);
        if d > radius {
            if d == 0.0 {
                continue;
            }
            let new_radius = (radius + d) * 0.5;
            let dir = (p - center) / d;
            center = center + dir * (new_radius - radius);
            radius = new_radius;
        }
    }
    let inflated = epsilon.max(0.0);
    Sphere {
        center,
        radius: radius + inflated,
    }
}

fn bounding_sphere(points: &[Vec3], epsilon: f32) -> Sphere {
    if points.is_empty() {
        return Sphere {
            center: Vec3::zero(),
            radius: 0.0,
        };
    }
    let mut min_x = points[0];
    let mut max_x = points[0];
    let mut min_y = points[0];
    let mut max_y = points[0];
    let mut min_z = points[0];
    let mut max_z = points[0];
    for &p in points.iter().skip(1) {
        if p.x < min_x.x {
            min_x = p;
        }
        if p.x > max_x.x {
            max_x = p;
        }
        if p.y < min_y.y {
            min_y = p;
        }
        if p.y > max_y.y {
            max_y = p;
        }
        if p.z < min_z.z {
            min_z = p;
        }
        if p.z > max_z.z {
            max_z = p;
        }
    }
    let seeds = [min_x, max_x, min_y, max_y, min_z, max_z];
    let mut best = ritter_sphere(points, seeds[0], epsilon);
    for &seed in seeds.iter().skip(1) {
        let candidate = ritter_sphere(points, seed, epsilon);
        if candidate.radius < best.radius {
            best = candidate;
        }
    }
    best
}

fn collect_points(tris: &[usize], triangles: &[Triangle], vertices: &[Vec3]) -> Vec<Vec3> {
    let mut points = Vec::with_capacity(tris.len() * 3);
    for &tri_idx in tris {
        let tri = &triangles[tri_idx];
        points.push(vertices[tri.indices[0]]);
        points.push(vertices[tri.indices[1]]);
        points.push(vertices[tri.indices[2]]);
    }
    points
}

fn cluster_sphere(
    tris: &[usize],
    triangles: &[Triangle],
    vertices: &[Vec3],
    epsilon: f32,
) -> Sphere {
    let points = collect_points(tris, triangles, vertices);
    bounding_sphere(&points, epsilon)
}

const CUT_RATIOS: [f32; 7] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

fn evaluate_best_split(
    tris: &[usize],
    triangles: &[Triangle],
    vertices: &[Vec3],
    parent_radius: f32,
    epsilon: f32,
) -> Option<SplitEval> {
    if tris.len() < 2 {
        return None;
    }
    let mut best: Option<SplitEval> = None;
    let len = tris.len();
    for axis in 0..3 {
        let mut sorted = tris.to_vec();
        sorted.sort_by(|&a, &b| {
            triangles[a]
                .centroid
                .axis(axis)
                .partial_cmp(&triangles[b].centroid.axis(axis))
                .unwrap_or(Ordering::Equal)
        });
        let mut last_mid: Option<usize> = None;
        for &ratio in CUT_RATIOS.iter() {
            let mid = ((len as f32) * ratio).round() as usize;
            if mid == 0 || mid >= len {
                continue;
            }
            if Some(mid) == last_mid {
                continue;
            }
            last_mid = Some(mid);
            let left_slice = &sorted[..mid];
            let right_slice = &sorted[mid..];
            let left_sphere = cluster_sphere(left_slice, triangles, vertices, epsilon);
            let right_sphere = cluster_sphere(right_slice, triangles, vertices, epsilon);
            let child_radius = left_sphere.radius.max(right_sphere.radius);
            let gain_abs = (parent_radius - child_radius).max(0.0);
            let gain_ratio = if parent_radius > 0.0 {
                (gain_abs / parent_radius).max(0.0)
            } else {
                0.0
            };
            let candidate = SplitEval {
                left_tris: left_slice.to_vec(),
                right_tris: right_slice.to_vec(),
                left_sphere,
                right_sphere,
                gain_ratio,
                gain_abs,
                cost: child_radius,
            };
            let should_take = match &best {
                None => true,
                Some(current) => {
                    candidate.cost < current.cost
                        || (candidate.cost == current.cost
                            && candidate.gain_ratio > current.gain_ratio)
                }
            };
            if should_take {
                best = Some(candidate);
            }
        }
    }
    best
}

fn spherize_mesh_impl(
    vertices: &[Vec3],
    triangles: &[Triangle],
    max_spheres: usize,
    min_gain_ratio: f32,
    min_gain_abs: f32,
    epsilon: f32,
) -> Vec<Sphere> {
    if triangles.is_empty() || max_spheres == 0 {
        return Vec::new();
    }
    let mut heap = BinaryHeap::new();
    let all_tris: Vec<usize> = (0..triangles.len()).collect();
    let root_sphere = cluster_sphere(&all_tris, triangles, vertices, epsilon);
    heap.push(Cluster {
        tris: all_tris,
        sphere: root_sphere,
    });

    let min_gain_ratio = min_gain_ratio.max(0.0);
    let min_gain_abs = min_gain_abs.max(0.0);
    let mut frozen: Vec<Cluster> = Vec::new();

    while heap.len() + frozen.len() < max_spheres {
        let Some(cluster) = heap.pop() else {
            break;
        };
        if cluster.tris.len() < 2 {
            frozen.push(cluster);
            continue;
        }
        let split = evaluate_best_split(
            &cluster.tris,
            triangles,
            vertices,
            cluster.sphere.radius,
            epsilon,
        );
        let Some(split) = split else {
            frozen.push(cluster);
            continue;
        };
        let threshold = min_gain_abs.max(min_gain_ratio * cluster.sphere.radius);
        if split.gain_abs < threshold {
            frozen.push(cluster);
            continue;
        }
        heap.push(Cluster {
            tris: split.left_tris,
            sphere: split.left_sphere,
        });
        heap.push(Cluster {
            tris: split.right_tris,
            sphere: split.right_sphere,
        });
    }

    let mut clusters = Vec::with_capacity(heap.len() + frozen.len());
    clusters.extend(heap.into_iter());
    clusters.extend(frozen);
    clusters.into_iter().map(|c| c.sphere).collect()
}

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from urdf-spherizer!".to_string()
}

#[pyfunction(signature = (
    vertices,
    indices,
    max_spheres,
    min_gain_ratio = None,
    epsilon = None,
    min_gain_abs = None
))]
fn spherize_mesh(
    vertices: PyReadonlyArray2<f32>,
    indices: PyReadonlyArray2<u32>,
    max_spheres: usize,
    min_gain_ratio: Option<f32>,
    epsilon: Option<f32>,
    min_gain_abs: Option<f32>,
) -> PyResult<Vec<([f32; 3], f32)>> {
    let vertices = vertices.as_array();
    let indices = indices.as_array();
    if vertices.ndim() != 2 || vertices.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "vertices must have shape (N, 3)",
        ));
    }
    if indices.ndim() != 2 || indices.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "indices must have shape (M, 3)",
        ));
    }
    if max_spheres == 0 {
        return Err(PyValueError::new_err("max_spheres must be >= 1"));
    }

    let mut verts = Vec::with_capacity(vertices.shape()[0]);
    for row in vertices.rows() {
        verts.push(Vec3 {
            x: row[0],
            y: row[1],
            z: row[2],
        });
    }

    let mut triangles = Vec::with_capacity(indices.shape()[0]);
    for row in indices.rows() {
        let i0 = row[0] as usize;
        let i1 = row[1] as usize;
        let i2 = row[2] as usize;
        if i0 >= verts.len() || i1 >= verts.len() || i2 >= verts.len() {
            return Err(PyValueError::new_err("indices out of bounds"));
        }
        let centroid = (verts[i0] + verts[i1] + verts[i2]) / 3.0;
        triangles.push(Triangle {
            indices: [i0, i1, i2],
            centroid,
        });
    }

    let min_gain_ratio = min_gain_ratio.unwrap_or(0.02);
    let epsilon = epsilon.unwrap_or(1.0e-6);
    let min_gain_abs = min_gain_abs.unwrap_or(0.0);
    let spheres = spherize_mesh_impl(
        &verts,
        &triangles,
        max_spheres,
        min_gain_ratio,
        min_gain_abs,
        epsilon,
    );
    Ok(spheres
        .into_iter()
        .map(|sphere| (sphere.center.into(), sphere.radius))
        .collect())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_bin, m)?)?;
    m.add_function(wrap_pyfunction!(spherize_mesh, m)?)?;
    Ok(())
}
