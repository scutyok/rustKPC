use cgmath::{Vector3, InnerSpace};
use std::io::Write;

// Reduce iterations to cut CPU cost; still iterative but cheaper.
const MAX_INTERSECT_PUSHBACK_ITERATIONS: usize = 12;
const EXTRA_PENETRATION_ADD: f32 = 0.02;
// Step-up parameters (stairs)
const STEP_HEIGHT: f32 = 0.9; // Match main.rs for reliable stepping and ground detection
const MAX_STEP_SLOPE_COS: f32 = 0.64; // ~50 degrees
const STEP_CLEARANCE: f32 = 0.02;

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
	pub min: Vector3<f32>,
	pub max: Vector3<f32>,
}

impl Aabb {
	pub fn contains(&self, point: Vector3<f32>) -> bool {
		point.x >= self.min.x && point.x <= self.max.x &&
		point.y >= self.min.y && point.y <= self.max.y &&
		point.z >= self.min.z && point.z <= self.max.z
	}

	pub fn intersects(&self, other: &Aabb) -> bool {
		!(self.max.x < other.min.x || self.min.x > other.max.x ||
		  self.max.y < other.min.y || self.min.y > other.max.y ||
		  self.max.z < other.min.z || self.min.z > other.max.z)
	}

	pub fn intersects_line_segment(&self, l0: Vector3<f32>, l1: Vector3<f32>) -> bool {
		let l = l1 - l0;
		let t = (l0 + l1) - (self.max + self.min);
		let e = self.max - self.min;

		if t.x.abs() > l.x.abs() + e.x { return false; }
		if t.y.abs() > l.y.abs() + e.y { return false; }
		if t.z.abs() > l.z.abs() + e.z { return false; }

		if (t.y * l.z - t.z * l.y).abs() > e.y * l.z.abs() + e.z * l.y.abs() { return false; }
		if (t.z * l.x - t.x * l.z).abs() > e.x * l.z.abs() + e.z * l.x.abs() { return false; }
		if (t.x * l.y - t.y * l.x).abs() > e.x * l.y.abs() + e.y * l.x.abs() { return false; }

		true
	}
}

mod bvh;

#[derive(Clone, Copy, Debug)]
pub struct PhysicsSphere {
	pub center: Vector3<f32>,
	pub radius: f32,
}

/// Height provider trait used by the engine to query ground height.
pub trait HeightProvider: Send + Sync {
	fn ground_height(&self, x: f32, y: f32, current_z: Option<f32>) -> f32;
}

/// Simple flat ground provider.
pub struct FlatGround;
impl HeightProvider for FlatGround {
	fn ground_height(&self, _x: f32, _y: f32, _current_z: Option<f32>) -> f32 { 0.0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerMode { Flying, Walk }

/// Ensure the player stays above ground (simple vertical correction).
pub fn resolve_player_collision(pos: &mut Vector3<f32>, height_provider: &dyn HeightProvider, player_radius: f32, _step_height: f32) {
	let ground_z = height_provider.ground_height(pos.x, pos.y, Some(pos.z));
	let min_z = ground_z + player_radius;
	// If below ground, snap to ground and avoid smoothing which causes oscillation.
	if pos.z < min_z {
		pos.z = min_z;
	}
}

#[derive(Clone)]
pub struct MeshHeightProvider {
	pub positions: Vec<Vector3<f32>>,
	pub indices: Vec<u32>,
	// Precomputed triangle centroids (xy) and XY radius for cheap early-out
	pub tri_centroids: Vec<Vector3<f32>>,
	pub tri_radius_xy: Vec<f32>,
	// Optional BVH for fast triangle queries
	pub bvh: Option<bvh::Bvh>,
}

impl MeshHeightProvider {
	pub fn new(positions: Vec<Vector3<f32>>, indices: Vec<u32>) -> Self {
		// Precompute per-triangle centroid and radius in XY
		let mut tri_centroids = Vec::new();
		let mut tri_radius_xy = Vec::new();
		let tri_count = indices.len() / 3;
		for t in 0..tri_count {
			let i0 = indices[t*3] as usize;
			let i1 = indices[t*3 + 1] as usize;
			let i2 = indices[t*3 + 2] as usize;
			if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
				tri_centroids.push(Vector3::new(0.0,0.0,0.0));
				tri_radius_xy.push(0.0);
				continue;
			}
			let a = positions[i0];
			let b = positions[i1];
			let c = positions[i2];
			let centroid = Vector3::new((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0, (a.z + b.z + c.z) / 3.0);
			let mut radius = 0.0f32;
			for v in [a, b, c].iter() {
				let dx = v.x - centroid.x;
				let dy = v.y - centroid.y;
				radius = radius.max((dx*dx + dy*dy).sqrt());
			}
			tri_centroids.push(centroid);
			tri_radius_xy.push(radius);
		}

		let bvh = if (indices.len() / 3) > 0 {
			Some(bvh::Bvh::new(&positions, &indices))
		} else { None };

		Self { positions, indices, tri_centroids, tri_radius_xy, bvh }
	}

	fn point_in_tri_2d(px: f32, py: f32, a: Vector3<f32>, b: Vector3<f32>, c: Vector3<f32>) -> bool {
		let v0x = c.x - a.x; let v0y = c.y - a.y;
		let v1x = b.x - a.x; let v1y = b.y - a.y;
		let v2x = px - a.x;  let v2y = py - a.y;

		let dot00 = v0x * v0x + v0y * v0y;
		let dot01 = v0x * v1x + v0y * v1y;
		let dot02 = v0x * v2x + v0y * v2y;
		let dot11 = v1x * v1x + v1y * v1y;
		let dot12 = v1x * v2x + v1y * v2y;

		let denom = dot00 * dot11 - dot01 * dot01;
		if denom.abs() < 1e-6 { return false; }
		let inv_denom = 1.0 / denom;
		let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
		let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
		(u >= 0.0) && (v >= 0.0) && (u + v <= 1.0)
	}

	fn setup_box(
		&self,
		p0: Vector3<f32>,
		p1: Vector3<f32>,
		dims: Vector3<f32>,
		offset: Vector3<f32>,
		radius: f32,
	) -> Option<(PhysicsSphere, PhysicsSphere, PhysicsSphere, Aabb)> {
		let v = p1 - p0;
		if v.magnitude2() < 0.0001 { return None; }

		let offset_pos0 = p0 + offset;
		let offset_pos1 = p1 + offset;

		let start_sphere = PhysicsSphere { center: offset_pos0, radius };
		let end_sphere = PhysicsSphere { center: offset_pos1, radius };
		let whole_sphere = PhysicsSphere { center: start_sphere.center + v * 0.5, radius: radius + v.magnitude() };

		let mut bmin = offset_pos0;
		let mut bmax = offset_pos0;
		bmin.x = bmin.x.min(offset_pos1.x);
		bmin.y = bmin.y.min(offset_pos1.y);
		bmin.z = bmin.z.min(offset_pos1.z);
		bmax.x = bmax.x.max(offset_pos1.x);
		bmax.y = bmax.y.max(offset_pos1.y);
		bmax.z = bmax.z.max(offset_pos1.z);

		bmin -= dims;
		bmax += dims;

		let boxa = Aabb { min: bmin, max: bmax };

		Some((start_sphere, end_sphere, whole_sphere, boxa))
	}

	/// More complete iterative wall-collision resolver based on the original engine.
	pub fn resolve_player_movement(&self, prev_pos: Vector3<f32>, pos: &mut Vector3<f32>, radius: f32) {
		// Make the player's collision box slightly smaller and less tall
		// (narrower horizontally and reduced vertical extent).
		let dims = Vector3::new(radius * 0.8, radius * 0.8, radius * 0.5);
		let offset = Vector3::new(0.0, 0.0, 0.0);

		let setup = self.setup_box(prev_pos, *pos, dims, offset, radius);
		if setup.is_none() { return; }
		let (_start_sphere, _end_sphere, _whole_sphere, boxa) = setup.unwrap();

		let idx = &self.indices;
		let posv = &self.positions;

		// gather candidate triangles via BVH if available
		let mut candidates: Vec<usize> = Vec::new();
		if let Some(ref b) = self.bvh {
			b.query_aabb(&boxa, &mut candidates);
		} else {
			candidates = (0..(idx.len() / 3)).collect();
		}

		for _iter in 0..MAX_INTERSECT_PUSHBACK_ITERATIONS {
			let mut moved = false;

			for &t in candidates.iter() {
				let i0 = idx[t*3] as usize;
				let i1 = idx[t*3 + 1] as usize;
				let i2 = idx[t*3 + 2] as usize;
				if i0 >= posv.len() || i1 >= posv.len() || i2 >= posv.len() { continue; }
				let a = posv[i0];
				let b = posv[i1];
				let c = posv[i2];

				// quick AABB check
				let tri_min = Vector3::new(a.x.min(b.x).min(c.x), a.y.min(b.y).min(c.y), a.z.min(b.z).min(c.z));
				let tri_max = Vector3::new(a.x.max(b.x).max(c.x), a.y.max(b.y).max(c.y), a.z.max(b.z).max(c.z));
				let tri_box = Aabb { min: tri_min - Vector3::new(radius, radius, radius), max: tri_max + Vector3::new(radius, radius, radius) };
				if !tri_box.intersects(&boxa) { continue; }

				// Cheap 2D centroid radius test to avoid expensive closest-point if far away
				let centroid = self.tri_centroids[t];
				let tri_radius = self.tri_radius_xy[t];
				let dx_c = pos.x - centroid.x;
				let dy_c = pos.y - centroid.y;
				let dist2 = dx_c*dx_c + dy_c*dy_c;
				let early_radius = (radius + tri_radius + 0.5) * (radius + tri_radius + 0.5);
				if dist2 > early_radius { continue; }

				let edge1 = b - a;
				let edge2 = c - a;
				let n = edge1.cross(edge2);

				// treat mostly-vertical triangles as walls (surface normal mostly horizontal)
				if n.z.abs() > 0.5 { continue; }

				// Attempt step-up (stairs) when this triangle blocks horizontal movement
				let tri_top_z = a.z.max(b.z).max(c.z);
				let foot_ground_z = self.ground_height(pos.x, pos.y, Some(pos.z));
				let delta = tri_top_z - foot_ground_z;
				let normal = {
					let nn = n;
					let mag = nn.magnitude();
					if mag > 1e-6 { nn / mag } else { nn }
				};
				let slope_ok = normal.z >= MAX_STEP_SLOPE_COS;
				if delta > 0.0 && delta <= STEP_HEIGHT && slope_ok {
					let mut stepped_pos = *pos;
					stepped_pos.z = foot_ground_z + delta + STEP_CLEARANCE;

					// For very small steps, skip headroom check entirely (for spiral/tight stairs)
					let skip_headroom = delta < 0.2;
					let headroom = if skip_headroom { 0.0 } else { 0.2 };
					let _head_clear = true;
					let head_top = stepped_pos.z + headroom;
					// Check for any triangle intersecting a vertical AABB above stepped_pos using BVH if available
					let mut head_clear = true;
					if !skip_headroom {
						let _head_top = stepped_pos.z + headroom;
						let head_aabb = Aabb {
							min: Vector3::new(stepped_pos.x - radius, stepped_pos.y - radius, stepped_pos.z),
							max: Vector3::new(stepped_pos.x + radius, stepped_pos.y + radius, head_top),
						};
						let mut head_overlaps: Vec<usize> = Vec::new();
						if let Some(ref bv) = self.bvh {
							bv.query_aabb(&head_aabb, &mut head_overlaps);
						} else {
							head_overlaps = (0..(idx.len() / 3)).collect();
						}
						for &ot in head_overlaps.iter() {
							if ot == t { continue; }
							let oi0 = idx[ot*3] as usize;
							let oi1 = idx[ot*3 + 1] as usize;
							let oi2 = idx[ot*3 + 2] as usize;
							if oi0 >= posv.len() || oi1 >= posv.len() || oi2 >= posv.len() { continue; }
							let oa = posv[oi0];
							let ob = posv[oi1];
							let oc = posv[oi2];
							let tri_min = Vector3::new(oa.x.min(ob.x).min(oc.x), oa.y.min(ob.y).min(oc.y), oa.z.min(ob.z).min(oc.z));
							let tri_max = Vector3::new(oa.x.max(ob.x).max(oc.x), oa.y.max(ob.y).max(oc.y), oa.z.max(ob.z).max(oc.z));
							let tri_box = Aabb { min: tri_min, max: tri_max };
							if head_aabb.intersects(&tri_box) {
								head_clear = false;
								break;
							}
						}
					}

					if delta < 0.2 {
						// For micro steps, if horizontal move is possible, allow step-up
						if stepped_pos.x != prev_pos.x || stepped_pos.y != prev_pos.y {
							pos.z = stepped_pos.z;
							if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("collision_log.txt") {
								let _ = writeln!(file, "[collision] step accepted t={} new_z={:.3}", t, pos.z);
							}
							continue;
						}
					}
					// For larger steps, do full step box collision check
					if delta >= 0.2 {
						let step_dims = dims;
						if let Some((_s0, _s1, _whole, step_box)) = self.setup_box(prev_pos, stepped_pos, step_dims, offset, radius) {
							let mut overlaps: Vec<usize> = Vec::new();
							if let Some(ref bv) = self.bvh {
								bv.query_aabb(&step_box, &mut overlaps);
							} else {
								overlaps = (0..(idx.len() / 3)).collect();
							}
							let mut blocked = false;
							for &ot in overlaps.iter() {
								// ignore the triangle we're attempting to step onto
								if ot == t { continue; }
								let oi0 = idx[ot*3] as usize;
								let oi1 = idx[ot*3 + 1] as usize;
								let oi2 = idx[ot*3 + 2] as usize;
								if oi0 >= posv.len() || oi1 >= posv.len() || oi2 >= posv.len() { continue; }
								let oa = posv[oi0];
								let ob = posv[oi1];
								let oc = posv[oi2];
								// full 3D distance from stepped sphere center to triangle
								let closest = closest_point_on_triangle(stepped_pos, oa, ob, oc);
								let diff = stepped_pos - closest;
								let dist = diff.magnitude();
								// if triangle penetrates sphere beyond a small epsilon, it's blocked
								if dist < (radius - 0.01) {
									blocked = true;
									// debug: log blocking triangle and metrics to file
									if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("collision_log.txt") {
										let _ = writeln!(file, "[collision] step blocked t={} dist={:.3} radius={:.3}", ot, dist, radius);
									}
									break;
								}
							}
							if !blocked && head_clear {
								pos.z = stepped_pos.z;
								if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("collision_log.txt") {
									let _ = writeln!(file, "[collision] step accepted t={} new_z={:.3}", t, pos.z);
								}
								continue;
							}
						}
					}
				}

				let closest = closest_point_on_triangle(*pos, a, b, c);

				let dx = pos.x - closest.x;
				let dy = pos.y - closest.y;
				let horiz_dist = (dx*dx + dy*dy).sqrt();

				// Reduce the vertical overlap buffer so the box is less tall
				let tri_min_z = a.z.min(b.z).min(c.z) - 0.2;
				let tri_max_z = a.z.max(b.z).max(c.z) + 0.2;
				if pos.z < tri_min_z || pos.z > tri_max_z { continue; }

				if horiz_dist < radius && horiz_dist > 1e-6 {
					let push = (radius - horiz_dist + EXTRA_PENETRATION_ADD).max(0.0);
					pos.x += dx / horiz_dist * push;
					pos.y += dy / horiz_dist * push;
					moved = true;
				} else if horiz_dist < 1e-6 {
					let mvx = pos.x - prev_pos.x;
					let mvy = pos.y - prev_pos.y;
					let mv_len = (mvx*mvx + mvy*mvy).sqrt();
					if mv_len > 1e-6 {
						pos.x += mvx / mv_len * (radius * 0.5);
						pos.y += mvy / mv_len * (radius * 0.5);
						moved = true;
					}
				}
			}

			if !moved { break; }
		}
	}
}

fn closest_point_on_triangle(p: Vector3<f32>, a: Vector3<f32>, b: Vector3<f32>, c: Vector3<f32>) -> Vector3<f32> {
	let ab = b - a;
	let ac = c - a;
	let ap = p - a;

	let d1 = ab.dot(ap);
	let d2 = ac.dot(ap);
	if d1 <= 0.0 && d2 <= 0.0 { return a; }

	let bp = p - b;
	let d3 = ab.dot(bp);
	let d4 = ac.dot(bp);
	if d3 >= 0.0 && d4 <= d3 { return b; }

	let vc = d1 * d4 - d3 * d2;
	if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
		let v = d1 / (d1 - d3);
		return a + ab * v;
	}

	let cp = p - c;
	let d5 = ab.dot(cp);
	let d6 = ac.dot(cp);
	if d6 >= 0.0 && d5 <= d6 { return c; }

	let vb = d5 * d2 - d1 * d6;
	if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
		let w = d2 / (d2 - d6);
		return a + ac * w;
	}

	let va = d3 * d6 - d5 * d4;
	if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
		let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + (c - b) * w;
	}

	let denom = 1.0 / (va + vb + vc);
	let v = vb * denom;
	let w = vc * denom;
	a + ab * v + ac * w
}

impl HeightProvider for MeshHeightProvider {
	fn ground_height(&self, x: f32, y: f32, current_z: Option<f32>) -> f32 {
		let mut best: Option<f32> = None;
		let idx = &self.indices;
		let pos = &self.positions;

		// query a small XY box around point to limit triangles
		let query_box = Aabb {
			min: Vector3::new(x - 1.0, y - 1.0, -1000.0),
			max: Vector3::new(x + 1.0, y + 1.0, 1000.0),
		};

		let mut candidates: Vec<usize> = Vec::new();
		if let Some(ref b) = self.bvh {
			b.query_aabb(&query_box, &mut candidates);
		} else {
			candidates = (0..(idx.len() / 3)).collect();
		}

		for &t in candidates.iter() {
			let i0 = idx[t*3] as usize;
			let i1 = idx[t*3 + 1] as usize;
			let i2 = idx[t*3 + 2] as usize;
			if i0 >= pos.len() || i1 >= pos.len() || i2 >= pos.len() { continue; }
			let v0 = pos[i0];
			let v1 = pos[i1];
			let v2 = pos[i2];

			if MeshHeightProvider::point_in_tri_2d(x, y, v0, v1, v2) {
				let edge1 = v1 - v0;
				let edge2 = v2 - v0;
				let n = edge1.cross(edge2);
				if n.z <= 0.0 { continue; }
				if n.z.abs() < 1e-6 { continue; }
				let z = v0.z - (n.x * (x - v0.x) + n.y * (y - v0.y)) / n.z;
				if let Some(curz) = current_z {
					if z > curz + 0.5 { continue; }
				}
				best = Some(best.map_or(z, |b| b.max(z)));
			}
		}
		best.unwrap_or(0.0)
	}
}


