pub type MatrixType<T> = Vec<Vec<T>>;

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

/// Reverse completemt of a DNA sequence (only 'A', 'C', 'T', 'G')
pub fn reverse_complement(seq: &str) -> String {
    seq.chars()
        .rev()
        .filter_map(|d| match d {
            'A' => Some('T'),
            'C' => Some('G'),
            'T' => Some('A'),
            'G' => Some('C'),
            _ => None,
        })
        .collect()
}
