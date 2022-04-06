use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[derive(Clone, Debug)]
struct SplitEval {
    left_tris: Vec<usize>,
    right_tris: Vec<usize>,
    left_sphere: Sphere,
    right_sphere: Sphere,
    gain_ratio: f32,
}

fn bounding_sphere(points: &[Vec3], epsilon: f32) -> Sphere {
    if points.is_empty() {
        return Sphere {
            center: Vec3::zero(),
            radius: 0.0,
        };
    }
    let mut p0 = points[0];
    let mut p1 = p0;
    let mut max_dist = 0.0_f32;
    for &p in points.iter().skip(1) {
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

fn split_cluster(tris: &[usize], triangles: &[Triangle]) -> Option<(Vec<usize>, Vec<usize>)> {
    if tris.len() < 2 {
        return None;
    }
    let mut min = triangles[tris[0]].centroid;
    let mut max = min;
    for &tri_idx in tris.iter().skip(1) {
        let c = triangles[tri_idx].centroid;
        min = min.min(c);
        max = max.max(c);
    }
    let range = max - min;
    let axis = if range.x >= range.y && range.x >= range.z {
        0
    } else if range.y >= range.z {
        1
    } else {
        2
    };
    let mut sorted = tris.to_vec();
    sorted.sort_by(|&a, &b| {
        triangles[a]
            .centroid
            .axis(axis)
            .partial_cmp(&triangles[b].centroid.axis(axis))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = sorted.len() / 2;
    if mid == 0 || mid == sorted.len() {
        return None;
    }
    let left = sorted[..mid].to_vec();
    let right = sorted[mid..].to_vec();
    Some((left, right))
}

fn evaluate_split(
    tris: &[usize],
    triangles: &[Triangle],
    vertices: &[Vec3],
    parent_radius: f32,
    epsilon: f32,
) -> Option<SplitEval> {
    let (left_tris, right_tris) = split_cluster(tris, triangles)?;
    let left_sphere = cluster_sphere(&left_tris, triangles, vertices, epsilon);
    let right_sphere = cluster_sphere(&right_tris, triangles, vertices, epsilon);
    let child_radius = left_sphere.radius.max(right_sphere.radius);
    if parent_radius <= 0.0 {
        return Some(SplitEval {
            left_tris,
            right_tris,
            left_sphere,
            right_sphere,
            gain_ratio: 0.0,
        });
    }
    let gain = parent_radius - child_radius;
    let gain_ratio = (gain / parent_radius).max(0.0);
    Some(SplitEval {
        left_tris,
        right_tris,
        left_sphere,
        right_sphere,
        gain_ratio,
    })
}

fn spherize_mesh_impl(
    vertices: &[Vec3],
    triangles: &[Triangle],
    max_spheres: usize,
    min_gain_ratio: f32,
    epsilon: f32,
) -> Vec<Sphere> {
    if triangles.is_empty() || max_spheres == 0 {
        return Vec::new();
    }
    let mut clusters = Vec::new();
    let all_tris: Vec<usize> = (0..triangles.len()).collect();
    let root_sphere = cluster_sphere(&all_tris, triangles, vertices, epsilon);
    clusters.push(Cluster {
        tris: all_tris,
        sphere: root_sphere,
    });

    let min_gain_ratio = min_gain_ratio.max(0.0);

    while clusters.len() < max_spheres {
        let mut best_index: Option<usize> = None;
        let mut best_split: Option<SplitEval> = None;
        let mut best_gain_ratio = min_gain_ratio;

        for (idx, cluster) in clusters.iter().enumerate() {
            if cluster.tris.len() < 2 {
                continue;
            }
            if let Some(split) = evaluate_split(
                &cluster.tris,
                triangles,
                vertices,
                cluster.sphere.radius,
                epsilon,
            ) {
                if split.gain_ratio > best_gain_ratio {
                    best_gain_ratio = split.gain_ratio;
                    best_index = Some(idx);
                    best_split = Some(split);
                }
            }
        }

        let Some(best_index) = best_index else {
            break;
        };
        let Some(best_split) = best_split else {
            break;
        };

        clusters.swap_remove(best_index);
        clusters.push(Cluster {
            tris: best_split.left_tris,
            sphere: best_split.left_sphere,
        });
        clusters.push(Cluster {
            tris: best_split.right_tris,
            sphere: best_split.right_sphere,
        });
    }

    clusters.into_iter().map(|c| c.sphere).collect()
}

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from urdf-spherizer!".to_string()
}

#[pyfunction]
fn spherize_mesh(
    vertices: PyReadonlyArray2<f32>,
    indices: PyReadonlyArray2<u32>,
    max_spheres: usize,
    min_gain_ratio: Option<f32>,
    epsilon: Option<f32>,
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
    let spheres = spherize_mesh_impl(&verts, &triangles, max_spheres, min_gain_ratio, epsilon);
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
