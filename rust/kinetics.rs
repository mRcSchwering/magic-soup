use rayon::prelude::*;

pub type DomainSpecType = ((u8, u8, u8, u8, u16), usize, usize);
pub type ProteinSpecType = (Vec<DomainSpecType>, usize, usize, bool);

pub trait Format {
    fn fmt_kwarg(&self) -> String;
}

enum Val {
    Int(usize),
    Str(String),
    Bool(bool),
    Arr(Vec<String>),
}

struct Kwarg {
    key: String,
    val: Val,
}

impl Format for Kwarg {
    fn fmt_kwarg(&self) -> String {
        let val = match &self.val {
            Val::Str(d) => format!("\"{d}\""),
            Val::Int(d) => format!("{d}"),
            Val::Bool(d) => format!("{d}"),
            Val::Arr(d) => format!("[{}]", d.join(",")),
        };
        format!("\"{}\":{}", self.key, val)
    }
}

fn get_protein(
    protein: &ProteinSpecType,
    vmaxs: &Vec<f32>,
    kms: &Vec<f32>,
    hills: &Vec<u8>,
    signs: &Vec<i8>,
    reacts: &Vec<Vec<i8>>,
    trnspts: &Vec<Vec<i8>>,
    effectors: &Vec<Vec<i8>>,
    molecules: &Vec<String>,
    n_signals: &usize,
) -> String {
    // yes this is ridiculous
    let domstrs: Vec<String> = (protein.0)
        .iter()
        .map(|(idxs, start, end)| {
            let start = format!("\"start\":{}", start);
            let end = format!("\"end\":{}", end);
            let kwargs = [format!("\"molecule\": \"{}\"", "asd")];
            let dom_kwargs = format!("{{{start},{end},{}}}", kwargs.join(","));
            format!("[{},{}]", idxs.0, dom_kwargs)
        })
        .collect();

    let kwargs: Vec<String> = [
        Kwarg {
            key: "cds_start".to_string(),
            val: Val::Int(protein.1),
        },
        Kwarg {
            key: "cds_end".to_string(),
            val: Val::Int(protein.2),
        },
        Kwarg {
            key: "is_fwd".to_string(),
            val: Val::Bool(protein.3),
        },
        Kwarg {
            key: "domains".to_string(),
            val: Val::Arr(domstrs),
        },
    ]
    .iter()
    .map(|d| d.fmt_kwarg())
    .collect();

    format!("{{{}}}", kwargs.join(","))
}

pub fn get_proteome_threaded(
    proteome: &Vec<ProteinSpecType>,
    vmaxs: &Vec<Vec<f32>>,
    kms: &Vec<Vec<f32>>,
    hills: &Vec<Vec<u8>>,
    signs: &Vec<Vec<i8>>,
    reacts: &Vec<Vec<Vec<i8>>>,
    trnspts: &Vec<Vec<Vec<i8>>>,
    effectors: &Vec<Vec<Vec<i8>>>,
    molecules: &Vec<String>,
) -> String {
    let n_signals = 2 * molecules.len();
    let proteins: Vec<String> = proteome
        .into_par_iter()
        .enumerate()
        .map(|(i, d)| {
            get_protein(
                &d,
                &vmaxs[i],
                &kms[i],
                &hills[i],
                &signs[i],
                &reacts[i],
                &trnspts[i],
                &effectors[i],
                molecules,
                &n_signals,
            )
        })
        .collect();

    format!("{{\"data\":[{}]}}", proteins.join(","))
}
