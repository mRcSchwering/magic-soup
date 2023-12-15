// Get vector with duplicates removed
pub fn unique<T: PartialEq + Clone>(mut pairs: Vec<T>) -> Vec<T> {
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

// Distance between a and b on circular 1D line of size m
pub fn dist_1d(a: &u16, b: &u16, m: &u16) -> u16 {
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

// Return coordinates of Moore's neighbourhood in square circular 2D map of size m
pub fn moores_nghbhd(x: &u16, y: &u16, m: &u16) -> [(u16, u16); 8] {
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

// For a position (x,y) get positions in Moore's neighborhood on circular 2D map of size map_size
// which are not already occupied as indicated by positions
pub fn free_moores_nghbhd(
    x: &u16,
    y: &u16,
    positions: &Vec<(u16, u16)>,
    map_size: &u16,
) -> Vec<(u16, u16)> {
    let nghbhd = moores_nghbhd(&x, &y, map_size);
    nghbhd
        .iter()
        .filter_map(|d| match positions.contains(d) {
            true => None,
            false => Some(*d),
        })
        .collect()
}
