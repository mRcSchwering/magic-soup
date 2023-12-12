pub static CODON_SIZE: usize = 3;

pub type DomainSpecType = ([usize; 5], usize, usize);
pub type ProteinSpecType = (Vec<DomainSpecType>, usize, usize, bool);
