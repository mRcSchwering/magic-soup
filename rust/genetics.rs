use rayon::prelude::*;
use std::collections::HashMap;
use std::str;

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
) -> Vec<(usize, usize, bool)> {
    let mut res: Vec<(usize, usize, bool)> = Vec::new();
    let n = seq.len();
    if n < *min_cds_size {
        return res;
    }

    // >99% expected <10 with 3 start and 3 stop codons
    let mut starts: [Vec<usize>; 3] = [
        Vec::with_capacity(15),
        Vec::with_capacity(15),
        Vec::with_capacity(15),
    ];

    for i in 0..(n - CODON_SIZE + 1) {
        let frame = i % CODON_SIZE;
        let j = i + CODON_SIZE;
        let codon = &seq[i..j];
        if start_codons.iter().any(|d| d == codon) {
            starts[frame].push(i);
        } else if stop_codons.iter().any(|d| d == codon) {
            while let Some(d) = starts[frame].pop() {
                if j - d >= *min_cds_size {
                    res.push((d, j, is_fwd))
                }
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
    genome: &str,
    cdss: &Vec<(usize, usize, bool)>,
    dom_size: &usize,
    dom_type_size: &usize,
    dom_type_map: &HashMap<String, usize>,
    one_codon_map: &HashMap<String, usize>,
    two_codon_map: &HashMap<String, usize>,
) -> Vec<ProteinSpecType> {
    let n_cdss = cdss.len();
    let mut res: Vec<ProteinSpecType> = Vec::with_capacity(n_cdss);
    if n_cdss == 0 {
        return res;
    }

    let i0e = CODON_SIZE;
    let i1e = 2 * CODON_SIZE;
    let i2e = 3 * CODON_SIZE;
    let i3e = 5 * CODON_SIZE;

    for (cds_start, cds_stop, is_fwd) in cdss {
        let n: usize = cds_stop - cds_start;
        let mut i: usize = 0;
        let mut is_useful_prot = false;

        // >99% < 3 with 3 start and 3 stop codons
        let mut doms: Vec<DomainSpecType> = Vec::with_capacity(4);

        while i + dom_size <= n {
            let dom_start = cds_start + i;
            let dom_type_end = dom_start + *dom_type_size;
            let dom_type_seq = &genome[dom_start..dom_type_end];

            if let Some(dom_type) = dom_type_map.get(dom_type_seq) {
                // 1=catal, 2=trnsp, 3=reg
                if *dom_type != (3 as usize) {
                    is_useful_prot = true;
                }
                let dom_spec_seq = &genome[dom_type_end..(dom_start + dom_size)];
                let i0 = one_codon_map.get(&dom_spec_seq[0..i0e]).unwrap();
                let i1 = one_codon_map.get(&dom_spec_seq[i0e..i1e]).unwrap();
                let i2 = one_codon_map.get(&dom_spec_seq[i1e..i2e]).unwrap();
                let i3 = two_codon_map.get(&dom_spec_seq[i2e..i3e]).unwrap();
                doms.push(([*dom_type, *i0, *i1, *i2, *i3], i, i + dom_size));
                i += *dom_size;
            } else {
                i += CODON_SIZE;
            }
        }

        // protein should have at least 1 non-regulatory domain
        if is_useful_prot {
            res.push((doms, *cds_start, *cds_stop, *is_fwd))
        }
    }
    res
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
fn translate_genome(
    genome: &str,
    start_codons: &Vec<String>,
    stop_codons: &Vec<String>,
    domain_map: &HashMap<String, usize>,
    one_codon_map: &HashMap<String, usize>,
    two_codon_map: &HashMap<String, usize>,
    dom_size: &usize,
    dom_type_size: &usize,
) -> Vec<ProteinSpecType> {
    let cds_fwd = get_coding_regions(genome, dom_size, start_codons, stop_codons, true);
    let mut doms_fwd = extract_domains(
        &genome,
        &cds_fwd,
        dom_size,
        dom_type_size,
        domain_map,
        one_codon_map,
        two_codon_map,
    );

    let bwd = reverse_complement(genome);
    let cds_bwd = get_coding_regions(&bwd, dom_size, start_codons, stop_codons, false);
    let mut doms_bwd = extract_domains(
        &bwd,
        &cds_bwd,
        dom_size,
        dom_type_size,
        domain_map,
        one_codon_map,
        two_codon_map,
    );

    doms_fwd.append(&mut doms_bwd);
    doms_fwd
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
                d,
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
