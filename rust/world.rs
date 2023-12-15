use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::cmp::PartialEq;

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

// 1D distance between a and b on circular dimension with size m
fn dist_1d(a: &u16, b: &u16, m: &u16) -> u16 {
    let mut h = b;
    let mut l = a;
    if a > b {
        h = a;
        l = b;
    }
    let d0 = h - l;
    let d1 = m - h + l;
    if d0 < d1 {
        return d0;
    }
    d1
}

// Return coordinates of Moore's neighbourhood in square circular map with size m
fn moores_nghbhd(x: &u16, y: &u16, m: &u16) -> [(u16, u16); 8] {
    let e: u16 = if x + 1 == *m { 0 } else { *x + 1 };
    let w: u16 = if x == &0 { *m - 1 } else { *x - 1 };
    let s: u16 = if y + 1 == *m { 0 } else { *y + 1 };
    let n: u16 = if y == &0 { *m - 1 } else { *y - 1 };
    [
        (w, n),
        (w, *y),
        (w, s),
        (*x, n),
        (*x, s),
        (e, n),
        (e, *y),
        (e, s),
    ]
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

// For a cell with idx cell_idx get all possible positions
// in Moore's neighborhood which are not already occupied
// as indicated by positions and a circular square 2D map of map_size
pub fn get_possible_cell_divisions(
    cell_idx: &usize,
    positions: &Vec<(u16, u16)>,
    map_size: &u16,
) -> Vec<(u16, u16)> {
    let (x, y) = positions[*cell_idx];
    let nghbhd = moores_nghbhd(&x, &y, map_size);
    nghbhd
        .iter()
        .filter_map(|d| match positions.contains(d) {
            true => None,
            false => Some(*d),
        })
        .collect()
}

// Threaded function that finds possible cell divisions for cell_idxs
// and divides them if they can as of get_possible_cell_divisions()
pub fn divide_cells_if_possible_threaded(
    cell_idxs: &Vec<usize>,
    positions: &Vec<(u16, u16)>,
    n_cells: &usize,
    map_size: &u16,
) -> (Vec<usize>, Vec<usize>, Vec<(u16, u16)>) {
    let opts: Vec<Vec<(u16, u16)>> = cell_idxs
        .into_par_iter()
        .map(|d| get_possible_cell_divisions(d, positions, map_size))
        .collect();

    let mut rng = rand::thread_rng();
    let mut running_idx = *n_cells;
    let mut succ_idxs: Vec<usize> = vec![];
    let mut child_idxs: Vec<usize> = vec![];
    let mut child_poss: Vec<(u16, u16)> = vec![];

    for (cell_idx, cell_opts) in cell_idxs.iter().zip(opts) {
        let cell_opts: Vec<&(u16, u16)> = cell_opts
            .iter()
            .filter(|d| !child_poss.contains(d))
            .collect();
        if let Some(&d) = cell_opts.choose(&mut rng) {
            let new_pos = *d;
            succ_idxs.push(*cell_idx);
            child_idxs.push(running_idx);
            child_poss.push(new_pos);
            running_idx += 1;
        }
    }

    (succ_idxs, child_idxs, child_poss)
}
