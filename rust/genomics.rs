extern crate pyo3;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;

pub static CODON_SIZE: usize = 3;

pub type CdsSpec = (usize, usize, bool);
pub type DomainTokens = (u8, u8, u8, u8, u16);
pub type DomainSpec = (DomainTokens, usize, usize);
pub type ProteinSpec = (Vec<DomainSpec>, usize, usize, bool);

/// Reverse completemt of a DNA sequence (only 'A', 'C', 'T', 'G')
pub fn reverse_complement(seq: &[u8]) -> Vec<u8> {
    seq.iter()
        .rev()
        .filter_map(|d| match d {
            b'A' => Some(b'T'),
            b'C' => Some(b'G'),
            b'T' => Some(b'A'),
            b'G' => Some(b'C'),
            _ => None,
        })
        .collect()
}

/// Find all CDSs in a circular genome using start and stop codons.
///
/// The genome is a ring, so a CDS may wrap past the end of the sequence and a
/// codon may straddle the seam. Scanning runs over an extended index range
/// `0..2n`: codons are read with wrap (`seq[i % n]`), reading frame follows the
/// extended index (`i % CODON_SIZE`) so it stays continuous across the seam for
/// any genome length, and starts are only opened in the first turn (`i < n`).
/// A start that finds no in-frame stop within one full turn would wrap forever,
/// so any CDS longer than `n` is dropped (same spirit as discarding un-stopped
/// CDSs). For a wrapped CDS the returned stop index exceeds `n`.
/// Each CDS has a minimum size of min_cds_size.
/// Returns each CDS with start and stop indices on genome and is_fwd.
pub fn get_coding_regions(
    seq: &[u8],
    min_cds_size: &usize,
    start_codons: &[[u8; CODON_SIZE]],
    stop_codons: &[[u8; CODON_SIZE]],
    is_53: bool,
) -> Vec<CdsSpec> {
    let n = seq.len();
    if n < *min_cds_size as usize {
        return Vec::with_capacity(0);
    }

    // >90% expected <15 CDSs per 1000 base pairs in one direction
    let mut res: Vec<CdsSpec> = Vec::with_capacity(15);

    // >99% expected <10 with 3 start and 3 stop codons
    let mut starts: [Vec<usize>; 3] = [
        Vec::with_capacity(12),
        Vec::with_capacity(12),
        Vec::with_capacity(12),
    ];

    // number of open starts; scanning the second turn stops once none remain
    let mut pending: usize = 0;

    for i in 0..(2 * n) {
        // first turn opened all starts; stop as soon as they are all closed
        if i >= n && pending == 0 {
            break;
        }

        let frame = i % CODON_SIZE;

        // codon with wrap across the seam
        let codon = if i + CODON_SIZE <= n {
            [seq[i], seq[i + 1], seq[i + 2]]
        } else {
            [seq[i % n], seq[(i + 1) % n], seq[(i + 2) % n]]
        };

        if i < n && start_codons.iter().any(|d| *d == codon) {
            starts[frame].push(i);
            pending += 1;
        } else if stop_codons.iter().any(|d| *d == codon) {
            let j = i + CODON_SIZE;
            while let Some(d) = starts[frame].pop() {
                pending -= 1;
                let len = j - d;
                // too long means the CDS wrapped a full turn without a stop
                if len >= *min_cds_size as usize && len <= n {
                    res.push((d, j, is_53));
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
    genome: &[u8],
    cdss: &Vec<CdsSpec>,
    dom_size: &usize,
    dom_type_size: &usize,
    domain_type_2_token: &HashMap<[u8; 2 * CODON_SIZE], u8>,
    one_codon_2_token: &HashMap<[u8; CODON_SIZE], u8>,
    two_codon_2_token: &HashMap<[u8; 2 * CODON_SIZE], u16>,
) -> Vec<ProteinSpec> {
    let n_cdss = cdss.len();
    if n_cdss == 0 {
        return Vec::with_capacity(0);
    }

    // A wrapped CDS (from a circular genome) has a stop index beyond the end of
    // the genome. Its domain regions are read from a once-doubled buffer so the
    // slices stay in bounds; non-wrapping genomes avoid the extra allocation.
    let n = genome.len();
    let buf: Cow<[u8]> = if cdss.iter().any(|(_, stop, _)| *stop > n) {
        Cow::Owned([genome, genome].concat())
    } else {
        Cow::Borrowed(genome)
    };
    let genome: &[u8] = &buf;

    // ofsets for domain regions
    let i0e = CODON_SIZE;
    let i1e = 2 * CODON_SIZE;
    let i2e = 3 * CODON_SIZE;
    let i3e = 5 * CODON_SIZE;

    // there will be <= n_cdss proteins
    let mut proteins: Vec<ProteinSpec> = Vec::with_capacity(n_cdss);

    // is_53 is only passed through
    for (cds_start, cds_stop, is_53) in cdss {
        let cds_size = cds_stop - cds_start;
        let mut is_useful_prot = false;

        // >99% < 3 with 3 start and 3 stop codons
        let mut domains: Vec<DomainSpec> = Vec::with_capacity(4);

        let mut i: usize = 0;
        while i + *dom_size <= cds_size {
            let dom_start = cds_start + i;
            let dom_type_end = dom_start + *dom_type_size;
            let dom_type_seq = &genome[dom_start..dom_type_end];

            // A domain is only valid if its type AND all four spec codons map to
            // tokens. start/stop codons are excluded from the codon maps, so a
            // lookup miss means there is no domain here. In particular, a domain
            // whose last codon coincides with the CDS's terminal stop codon has
            // an unmapped two-codon spec and is correctly rejected. On any miss
            // advance a single codon and retry (like a domain-type non-match).
            let dom_end = dom_start + *dom_size;
            let dom_spec_seq = &genome[dom_type_end..dom_end];
            match (
                domain_type_2_token.get(dom_type_seq),
                one_codon_2_token.get(&dom_spec_seq[0..i0e]),
                one_codon_2_token.get(&dom_spec_seq[i0e..i1e]),
                one_codon_2_token.get(&dom_spec_seq[i1e..i2e]),
                two_codon_2_token.get(&dom_spec_seq[i2e..i3e]),
            ) {
                (Some(dom_type_token), Some(t0), Some(t1), Some(t2), Some(t3)) => {
                    // 1=catal, 2=trnsp, 3=reg
                    if *dom_type_token != 3 {
                        is_useful_prot = true;
                    }
                    domains.push(((*dom_type_token, *t0, *t1, *t2, *t3), i, i + *dom_size));
                    i += *dom_size;
                }
                _ => i += CODON_SIZE,
            }
        }

        // protein should have at least 1 non-regulatory domain
        if is_useful_prot {
            proteins.push((domains, *cds_start, *cds_stop, *is_53))
        }
    }
    proteins
}

/// For a genome, extract CDSs on forward and reverse-complement,
/// then extract protein specification for each CDS and return them.
fn translate_genome(
    genome: &[u8],
    start_codons: &[[u8; CODON_SIZE]],
    stop_codons: &[[u8; CODON_SIZE]],
    domain_type_2_token: &HashMap<[u8; 2 * CODON_SIZE], u8>,
    one_codon_2_token: &HashMap<[u8; CODON_SIZE], u8>,
    two_codon_2_token: &HashMap<[u8; 2 * CODON_SIZE], u16>,
    dom_size: &usize,
    dom_type_size: &usize,
    min_cds_size: &usize,
) -> Vec<ProteinSpec> {
    let cds_fwd = get_coding_regions(genome, min_cds_size, start_codons, stop_codons, true);
    let mut doms_fwd = extract_domains(
        &genome,
        &cds_fwd,
        dom_size,
        dom_type_size,
        domain_type_2_token,
        one_codon_2_token,
        two_codon_2_token,
    );

    let bwd = reverse_complement(genome);
    let cds_bwd = get_coding_regions(&bwd, min_cds_size, start_codons, stop_codons, false);
    let mut doms_bwd = extract_domains(
        &bwd,
        &cds_bwd,
        dom_size,
        dom_type_size,
        domain_type_2_token,
        one_codon_2_token,
        two_codon_2_token,
    );

    doms_fwd.append(&mut doms_bwd);
    doms_fwd
}

// Threaded version of translate_genome() for multiple genomes
pub fn translate_genomes_threaded(
    genomes: &[&[u8]],
    start_codons: &[[u8; CODON_SIZE]],
    stop_codons: &[[u8; CODON_SIZE]],
    domain_type_2_token: &HashMap<[u8; 2 * CODON_SIZE], u8>,
    one_codon_2_token: &HashMap<[u8; CODON_SIZE], u8>,
    two_codon_2_token: &HashMap<[u8; 2 * CODON_SIZE], u16>,
    dom_size: &usize,
    dom_type_size: &usize,
    min_cds_size: &usize,
) -> Vec<Vec<ProteinSpec>> {
    genomes
        .into_par_iter()
        .map(|d| {
            translate_genome(
                d,
                start_codons,
                stop_codons,
                domain_type_2_token,
                one_codon_2_token,
                two_codon_2_token,
                dom_size,
                dom_type_size,
                min_cds_size,
            )
        })
        .collect()
}
