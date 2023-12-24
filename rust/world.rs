use crate::util;
use rand::seq::SliceRandom;
use rayon::prelude::*;

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
    // maximum 8
    let mut res: Vec<(usize, usize)> = Vec::with_capacity(8);

    let (x0, y0) = positions[*from_idx];

    for to_idx in to_idxs {
        if to_idx == from_idx {
            continue;
        }

        let (x1, y1) = positions[*to_idx];
        let x_d = util::dist_1d(&x0, &x1, map_size);
        let y_d = util::dist_1d(&y0, &y1, map_size);
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

    util::unique(res)
}

// Threaded function that finds possible cell divisions as of
// free_moores_nghbhd() for multiple cells as cell_idxs
// Returns divided cell idxs, their child idx, their child positions
pub fn divide_cells_if_possible_threaded(
    cell_idxs: &Vec<usize>,
    positions: &Vec<(u16, u16)>,
    n_cells: &usize,
    map_size: &u16,
) -> (Vec<usize>, Vec<usize>, Vec<(u16, u16)>) {
    let opts: Vec<Vec<(u16, u16)>> = cell_idxs
        .into_par_iter()
        .map(|d| {
            let (x, y) = positions[*d];
            util::free_moores_nghbhd(&x, &y, positions, map_size)
        })
        .collect();

    let mut rng = rand::thread_rng();

    // maximum n dividing cells
    let n_dividing_cells = cell_idxs.len();
    let mut succ_idxs: Vec<usize> = Vec::with_capacity(n_dividing_cells);
    let mut child_idxs: Vec<usize> = Vec::with_capacity(n_dividing_cells);
    let mut child_poss: Vec<(u16, u16)> = Vec::with_capacity(n_dividing_cells);

    let mut running_idx = *n_cells;
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

// Threaded function that finds possible cell movements as of
// free_moores_nghbhd() for multiple cells as cell_idxs
// Returns new position tuples, moved cell idxs
pub fn move_cells_threaded(
    cell_idxs: &Vec<usize>,
    positions: &Vec<(u16, u16)>,
    map_size: &u16,
) -> (Vec<(u16, u16)>, Vec<usize>) {
    let const_cell_idxs: Vec<usize> = (0..positions.len())
        .filter_map(|d| match cell_idxs.binary_search(&d) {
            Ok(_) => None,
            _ => Some(d),
        })
        .collect();

    let const_occ_pos: Vec<(u16, u16)> = const_cell_idxs.iter().map(|d| positions[*d]).collect();
    let opts: Vec<Vec<(u16, u16)>> = cell_idxs
        .into_par_iter()
        .map(|d| {
            let (x, y) = positions[*d];
            util::free_moores_nghbhd(&x, &y, &const_occ_pos, map_size)
        })
        .collect();

    let mut rng = rand::thread_rng();

    // maximum n moving cells
    let n_moving_cells = cell_idxs.len();
    let mut moved_idxs: Vec<usize> = Vec::with_capacity(n_moving_cells);
    let mut new_positions: Vec<(u16, u16)> = Vec::with_capacity(n_moving_cells);
    let mut current_occ_pos: Vec<(u16, u16)> = cell_idxs.iter().map(|d| positions[*d]).collect();

    for (idx, cell_idx) in cell_idxs.iter().enumerate() {
        let cell_opts: Vec<&(u16, u16)> = opts[idx]
            .iter()
            .filter(|d| !current_occ_pos.contains(d))
            .collect();

        if let Some(&d) = cell_opts.choose(&mut rng) {
            let new_pos = *d;
            new_positions.push(new_pos);
            moved_idxs.push(*cell_idx);
            current_occ_pos[idx] = new_pos;
        }
    }

    (new_positions, moved_idxs)
}
