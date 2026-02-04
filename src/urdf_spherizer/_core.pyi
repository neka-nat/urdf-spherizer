def hello_from_bin() -> str: ...
def spherize_mesh(
    vertices,
    indices,
    max_spheres: int,
    min_gain_ratio: float | None = ...,
    epsilon: float | None = ...,
    min_gain_abs: float | None = ...,
) -> list[tuple[tuple[float, float, float], float]]: ...
