use super::Aabb;
use cgmath::Vector3;

#[derive(Clone)]
pub struct Bvh {
    pub nodes: Vec<Node>,
    pub tri_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub aabb: Aabb,
    pub start: usize,
    pub count: usize,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

impl Bvh {
    pub fn new(positions: &[Vector3<f32>], indices: &[u32]) -> Self {
        let tri_count = indices.len() / 3;
        let mut tri_indices: Vec<usize> = (0..tri_count).collect();
        let mut nodes: Vec<Node> = Vec::new();
        let mut tri_storage: Vec<usize> = Vec::with_capacity(tri_count);

        fn compute_tri_aabb(positions: &[Vector3<f32>], indices: &[u32], t: usize) -> Aabb {
            let i0 = indices[t*3] as usize;
            let i1 = indices[t*3 + 1] as usize;
            let i2 = indices[t*3 + 2] as usize;
            let a = positions[i0];
            let b = positions[i1];
            let c = positions[i2];
            let min = Vector3::new(a.x.min(b.x).min(c.x), a.y.min(b.y).min(c.y), a.z.min(b.z).min(c.z));
            let max = Vector3::new(a.x.max(b.x).max(c.x), a.y.max(b.y).max(c.y), a.z.max(b.z).max(c.z));
            Aabb { min, max }
        }

        fn merge_aabb(a: &Aabb, b: &Aabb) -> Aabb {
            Aabb { min: Vector3::new(a.min.x.min(b.min.x), a.min.y.min(b.min.y), a.min.z.min(b.min.z)),
                   max: Vector3::new(a.max.x.max(b.max.x), a.max.y.max(b.max.y), a.max.z.max(b.max.z)) }
        }

        fn build_recursive(
            positions: &[Vector3<f32>],
            indices: &[u32],
            tri_indices: &mut [usize],
            nodes: &mut Vec<Node>,
            tri_storage: &mut Vec<usize>,
        ) -> usize {
            // compute node aabb
            let mut node_aabb: Option<Aabb> = None;
            for &ti in tri_indices.iter() {
                let aabb = compute_tri_aabb(positions, indices, ti);
                node_aabb = Some(match node_aabb {
                    None => aabb,
                    Some(prev) => merge_aabb(&prev, &aabb),
                });
            }
            let node_aabb = node_aabb.unwrap_or(Aabb { min: Vector3::new(0.0,0.0,0.0), max: Vector3::new(0.0,0.0,0.0) });

            let node_index = nodes.len();
            nodes.push(Node { aabb: node_aabb.clone(), start: 0, count: 0, left: None, right: None });

            // leaf threshold
            if tri_indices.len() <= 8 {
                let start = tri_storage.len();
                for &ti in tri_indices.iter() { tri_storage.push(ti); }
                let count = tri_indices.len();
                nodes[node_index].start = start;
                nodes[node_index].count = count;
                return node_index;
            }

            // choose split axis by extent
            let ext_x = node_aabb.max.x - node_aabb.min.x;
            let ext_y = node_aabb.max.y - node_aabb.min.y;
            let ext_z = node_aabb.max.z - node_aabb.min.z;
            let axis = if ext_x >= ext_y && ext_x >= ext_z { 0 } else if ext_y >= ext_x && ext_y >= ext_z { 1 } else { 2 };

            // compute centroid sort
            tri_indices.sort_unstable_by(|&a, &b| {
                let ia0 = indices[a*3] as usize; let ia1 = indices[a*3 + 1] as usize; let ia2 = indices[a*3 + 2] as usize;
                let ba0 = indices[b*3] as usize; let ba1 = indices[b*3 + 1] as usize; let ba2 = indices[b*3 + 2] as usize;
                let ca = (positions[ia0] + positions[ia1] + positions[ia2]) * (1.0/3.0);
                let cb = (positions[ba0] + positions[ba1] + positions[ba2]) * (1.0/3.0);
                let va = if axis == 0 { ca.x } else if axis == 1 { ca.y } else { ca.z };
                let vb = if axis == 0 { cb.x } else if axis == 1 { cb.y } else { cb.z };
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mid = tri_indices.len() / 2;
            let (left_slice, right_slice) = tri_indices.split_at_mut(mid);
            let left = build_recursive(positions, indices, left_slice, nodes, tri_storage);
            let right = build_recursive(positions, indices, right_slice, nodes, tri_storage);
            nodes[node_index].left = Some(left);
            nodes[node_index].right = Some(right);
            nodes[node_index].start = 0;
            nodes[node_index].count = 0;
            node_index
        }

        // build
        if tri_count == 0 {
            return Bvh { nodes, tri_indices: tri_storage };
        }
        build_recursive(positions, indices, &mut tri_indices[..], &mut nodes, &mut tri_storage);
        Bvh { nodes, tri_indices: tri_storage }
    }

    pub fn query_aabb(&self, aabb: &Aabb, out: &mut Vec<usize>) {
        if self.nodes.is_empty() { return; }
        let mut stack: Vec<usize> = Vec::new();
        stack.push(0);
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            if !node.aabb.intersects(aabb) { continue; }
            if node.count > 0 {
                for i in 0..node.count {
                    out.push(self.tri_indices[node.start + i]);
                }
            } else {
                if let Some(r) = node.right { stack.push(r); }
                if let Some(l) = node.left { stack.push(l); }
            }
        }
    }
}
