use rayon::prelude::*;
use std::cmp::PartialEq;

// 1D distance between a and b on circular dimension with size s
fn dist_1d(a: &u16, b: &u16, s: &u16) -> u16 {
    let mut h = b;
    let mut l = a;
    if a > b {
        h = a;
        l = b;
    }
    let d0 = h - l;
    let d1 = s - h + l;
    if d0 < d1 {
        return d0;
    }
    d1
}

// get all neighbours of cell from_idx to cells to_idxs
// on a square circular 2D map with size map_size
// Positions of each cell are given as x,y tuples in positions.
// Returns tuples of 2 neighbouring indexes, smaller idx first.
fn get_neighbors(
    from_idx: &usize,
    to_idxs: &Vec<usize>,
    positions: &Vec<(u16, u16)>,
    map_size: &u16,
) -> Vec<(usize, usize)> {
    let mut res: Vec<(usize, usize)> = Vec::new();

    let (x0, y0) = positions[*from_idx];

    for to_idx in to_idxs {
        if to_idx == from_idx {
            continue;
        }

        let (x1, y1) = positions[*to_idx];
        let x_d = dist_1d(&x0, &x1, map_size);
        let y_d = dist_1d(&y0, &y1, map_size);
        if x_d <= 1 && y_d <= 1 {
            if from_idx > to_idx {
                res.push((*to_idx, *from_idx));
            } else {
                res.push((*from_idx, *to_idx));
            }
        }
    }

    res
}

// Get vector with duplicates removed
fn unique<T: PartialEq + Clone>(mut pairs: Vec<T>) -> Vec<T> {
    let mut seen: Vec<T> = Vec::with_capacity(pairs.len());
    pairs.retain(|d| match seen.contains(d) {
        true => false,
        _ => {
            seen.push(d.clone());
            true
        }
    });
    pairs
}

// Threaded version of get_neighbors() for multiple from_idxs
pub fn get_neighbors_threaded(
    from_idxs: &Vec<usize>,
    to_idxs: &Vec<usize>,
    positions: &Vec<(u16, u16)>,
    map_size: &u16,
) -> Vec<(usize, usize)> {
    let res: Vec<(usize, usize)> = from_idxs
        .into_par_iter()
        .map(|d| get_neighbors(&d, to_idxs, positions, map_size))
        .flatten()
        .collect();

    unique(res)
}

fn moores_nghbhd(x: &u16, y: &u16, size: &u16) -> [(u16, u16); 8] {
    let cardinals: [(i16, i16); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    cardinals.map(|(dx, dy)| ((*x as i16 + dx) as u16, (*y as i16 + dy) as u16))
}

pub fn divide_cells_if_possible(
    cell_idxs: &Vec<usize>,
    positions: &Vec<(u16, u16)>,
    n_cells: &usize,
    map_size: &u16,
) -> (Vec<usize>, Vec<usize>, Vec<(u16, u16)>) {
    let mut rng_idx = n_cells;
    let mut succ_idxs: Vec<usize> = vec![];
    let mut child_idxs: Vec<usize> = vec![];
    let mut child_poss: Vec<(u16, u16)> = vec![];

    (succ_idxs, child_idxs, child_poss)
}
