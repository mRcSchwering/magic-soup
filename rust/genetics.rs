use rayon::prelude::*;
use std::collections::HashMap;

static CODON_SIZE: usize = 3;

pub type DomainSpecType = ([usize; 5], usize, usize);
pub type ProteinSpecType = (Vec<DomainSpecType>, usize, usize, bool);

/// Find all CDSs in genome using start and stop codons.
/// Each CDS has a minimum size of min_cds_size.
/// Returns each CDS with start and stop indices on genome and is_fwd.
pub fn get_coding_regions(
    seq: &str,
    min_cds_size: &usize,
    start_codons: &Vec<String>,
    stop_codons: &Vec<String>,
    is_fwd: bool,
) -> Vec<(String, usize, usize, bool)> {
    let mut res: Vec<(String, usize, usize, bool)> = Vec::new();
    let n = seq.len();
    if n < *min_cds_size {
        return res;
    }

    let mut start_idxs: Vec<usize> = Vec::new();
    for start_codon in start_codons {
        let idxs: Vec<usize> = seq[..(n - *min_cds_size + CODON_SIZE)]
            .match_indices(start_codon)
            .map(|(i, _)| i)
            .collect();
        start_idxs.extend(idxs);
    }

    let mut stop_idxs: Vec<usize> = Vec::new();
    for stop_codon in stop_codons {
        let idxs: Vec<usize> = seq[..n].match_indices(stop_codon).map(|(i, _)| i).collect();
        stop_idxs.extend(idxs);
    }

    start_idxs.sort();
    stop_idxs.sort();

    let mut f0_starts: Vec<usize> = Vec::new();
    let mut f1_starts: Vec<usize> = Vec::new();
    let mut f2_starts: Vec<usize> = Vec::new();
    let mut f0_stops: Vec<usize> = Vec::new();
    let mut f1_stops: Vec<usize> = Vec::new();
    let mut f2_stops: Vec<usize> = Vec::new();
    for start_idx in start_idxs {
        match start_idx % 3 {
            0 => f0_starts.push(start_idx),
            1 => f1_starts.push(start_idx),
            2 => f2_starts.push(start_idx),
            _ => {}
        }
    }
    for stop_idx in stop_idxs {
        match stop_idx % 3 {
            0 => f0_stops.push(stop_idx),
            1 => f1_stops.push(stop_idx),
            2 => f2_stops.push(stop_idx),
            _ => {}
        }
    }

    let frames = [
        (&f0_starts, &mut f0_stops),
        (&f1_starts, &mut f1_stops),
        (&f2_starts, &mut f2_stops),
    ];

    for (f_starts, f_stops) in frames {
        for f_start in f_starts {
            f_stops.retain(|d| d > &(f_start + CODON_SIZE));
            let stop0 = f_stops.iter().min();
            if let Some(d) = stop0 {
                let f_stop = d + CODON_SIZE;
                if f_stop - f_start > *min_cds_size {
                    let cds = seq[*f_start..f_stop].to_string();
                    res.push((cds, *f_start, f_stop, is_fwd))
                }
            } else {
                break;
            }
        }
    }

    res
}

/// Extract domain specification from a list of CDSs.
/// Domains are defined by DNA regions which map to domain type and indices.
/// Mappings are defined by HashMaps, sizes by ints.
/// Returns list of protein specifications, which are in turn each a list with
/// domain specifications with each specification indices, start/stop indices,
/// and strand direction information.
pub fn extract_domains(
    cdss: &Vec<(String, usize, usize, bool)>,
    dom_size: &usize,
    dom_type_size: &usize,
    dom_type_map: &HashMap<String, usize>,
    one_codon_map: &HashMap<String, usize>,
    two_codon_map: &HashMap<String, usize>,
) -> Vec<ProteinSpecType> {
    let i0s = 0..CODON_SIZE;
    let i1s = CODON_SIZE..(2 * CODON_SIZE);
    let i2s = (2 * CODON_SIZE)..(3 * CODON_SIZE);
    let i3s = (3 * CODON_SIZE)..(5 * CODON_SIZE);
    let mut prot_doms: Vec<ProteinSpecType> = Vec::new();

    for (cds, cds_start, cds_stop, is_fwd) in cdss {
        let n: usize = cds.len();
        let mut i: usize = 0;
        let mut is_useful_prot = false;
        let mut doms: Vec<DomainSpecType> = Vec::new();

        while i + dom_size <= n {
            let dom_type_seq = cds[i..(i + *dom_type_size)].to_string();
            let dom_type_opt = dom_type_map.get(&dom_type_seq);

            if let Some(dom_type) = dom_type_opt {
                // 1=catal, 2=trnsp, 3=reg
                if *dom_type != (3 as usize) {
                    is_useful_prot = true;
                }
                let dom_spec_seq = cds[(i + dom_type_size)..(i + dom_size)].to_string();
                let i0 = one_codon_map.get(&dom_spec_seq[i0s.clone()]).unwrap();
                let i1 = one_codon_map.get(&dom_spec_seq[i1s.clone()]).unwrap();
                let i2 = one_codon_map.get(&dom_spec_seq[i2s.clone()]).unwrap();
                let i3 = two_codon_map.get(&dom_spec_seq[i3s.clone()]).unwrap();
                doms.push(([*dom_type, *i0, *i1, *i2, *i3], i, i + dom_size));
                i += *dom_size;
            } else {
                i += CODON_SIZE;
            }
        }

        // protein should have at least 1 non-regulatory domain
        if is_useful_prot {
            prot_doms.push((doms, *cds_start, *cds_stop, *is_fwd))
        }
    }
    prot_doms
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

/// For a genome, extract CDSs on forward and reverse-complement,
/// then extract protein specification for each CDS and return them.
pub fn translate_genome(
    genome: &str,
    start_codons: &Vec<String>,
    stop_codons: &Vec<String>,
    domain_map: &HashMap<String, usize>,
    one_codon_map: &HashMap<String, usize>,
    two_codon_map: &HashMap<String, usize>,
    dom_size: &usize,
    dom_type_size: &usize,
) -> Vec<ProteinSpecType> {
    let mut cdsf = get_coding_regions(genome, dom_size, start_codons, stop_codons, true);

    let bwd = reverse_complement(genome);
    let mut cdsb = get_coding_regions(&bwd, dom_size, start_codons, stop_codons, false);

    cdsf.append(&mut cdsb);

    extract_domains(
        &cdsf,
        dom_size,
        dom_type_size,
        domain_map,
        one_codon_map,
        two_codon_map,
    )
}

// Threaded version of translate_genome() for multiple genomes
pub fn translate_genomes(
    genomes: &Vec<String>,
    start_codons: &Vec<String>,
    stop_codons: &Vec<String>,
    domain_map: &HashMap<String, usize>,
    one_codon_map: &HashMap<String, usize>,
    two_codon_map: &HashMap<String, usize>,
    dom_size: &usize,
    dom_type_size: &usize,
) -> Vec<Vec<ProteinSpecType>> {
    genomes
        .into_par_iter()
        .map(|d| {
            translate_genome(
                &d,
                start_codons,
                stop_codons,
                domain_map,
                one_codon_map,
                two_codon_map,
                dom_size,
                &dom_type_size,
            )
        })
        .collect()
}
