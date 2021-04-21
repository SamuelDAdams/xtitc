mod util;
use std::{error::Error};
use util::*;

#[derive(Default, Clone)]
pub struct Context {
    pub instance_count: usize,
    pub class_label_count: usize,
    pub original_attr_count: usize,
    pub attribute_count: usize,
    pub feature_count: usize,
    pub bin_count: usize,
    pub tree_count: usize,
    pub max_depth: usize,
    pub epsilon: f64,
    pub decimal_precision: usize,
    pub seed: usize,
}

#[derive(Default, Clone)]
pub struct Node {
    pub attribute: usize,
    pub value: f64,
    pub frequencies: Vec<usize>,
}

fn main() {
    let fileloc = "settings/settings1.toml";
    //load settings
    let (ctx, data, classes) = init(&fileloc.to_string()).unwrap();
    let use_frequencies = false;
    let (disc_data, feature_selectors, feature_values) = xt_preprocess(&data, &ctx).unwrap();
    let trees = sid3t(&disc_data, &feature_selectors, &feature_values, &ctx);
    //preprocess dataset according to the settings
    
}

pub fn sid3t(data: &Vec<Vec<Vec<usize>>>, subset_indices: &Vec<Vec<usize>>, split_points: &Vec<Vec<f64>>, ctx: &Context) -> Result<Vec<Vec<Node>>, Box<dyn Error>>{
    let feature_count = ctx.feature_count;
    let max_depth = ctx.max_depth;
    let epsilon = ctx.epsilon;
    let tree_count = ctx.tree_count;
    let instance_count = ctx.instance_count;

    let trees = vec![vec![Node {
        attribute: 0,
        value: 0f64,
        frequencies: vec![],
    }]; tree_count];

    let transaction_subsets = vec![vec![vec![1usize; instance_count]]; ctx.tree_count]; //3d treecount x nodes_to_process_per_tree x instance_count
    let ances_class_bits = 
    for d in 0 .. max_depth {

    }

    Ok(trees)
}

pub fn init(cfg_file: &String) -> Result<(Context, Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn Error>> {
	let mut settings = config::Config::default();
    settings
        .merge(config::File::with_name(cfg_file.as_str())).unwrap()
        .merge(config::Environment::with_prefix("APP")).unwrap();

    let class_label_count: usize = settings.get_int("class_label_count")? as usize;
    let attribute_count: usize = settings.get_int("attribute_count")? as usize;
    let instance_count: usize = settings.get_int("instance_count")? as usize;
    let feature_count: usize = settings.get_int("feature_count")? as usize;
    let tree_count: usize = settings.get_int("tree_count")? as usize;
    let max_depth: usize = settings.get_int("max_depth")? as usize;
    let seed: usize = settings.get_int("seed")? as usize;
    let epsilon: f64 = settings.get_int("epsilon")? as f64;
    let decimal_precision: usize = settings.get_int("decimal_precision")? as usize;
    let original_attr_count = attribute_count;
    let bin_count = 2usize;

    let data = matrix_csv_to_float_vec(&settings.get_str("data")?)?;
    let data = data.iter().map(|x| x.iter().map(|y| truncate(y, decimal_precision).unwrap()).collect()).collect();
    let mut classes = matrix_csv_to_float_vec(&settings.get_str("classes")?)?;

    classes = transpose(&classes)?;
    classes = classes.iter().map(|x| x.iter().map(|y| truncate(y, decimal_precision).unwrap()).collect()).collect();


    let c = Context {
        instance_count,
        class_label_count,
        attribute_count,
        feature_count,
        original_attr_count,
        bin_count,
        tree_count,
        max_depth,
        epsilon,
        decimal_precision,
        seed,
    };

    Ok((c, data, classes))
}

pub fn xt_preprocess(data: &Vec<Vec<f64>>, ctx: &Context) -> Result<(Vec<Vec<Vec<usize>>>, Vec<Vec<usize>>, Vec<Vec<f64>>), Box<dyn Error>>{
    let maxes: Vec<f64> = data.iter().map(|x| x.iter().cloned().fold(0./0., f64::max)).collect();
    let mins: Vec<f64> = data.iter().map(|x| x.iter().cloned().fold(1./0., f64::min)).collect();
    let ratios = get_ratios(ctx.feature_count * ctx.tree_count, ctx.decimal_precision, ctx.seed)?;
    let ranges: Vec<f64> = maxes.iter().zip(mins.iter()).map(|(max , min)| max - min).collect();
    let features = get_features(ctx.feature_count * ctx.tree_count, ctx.attribute_count, ctx.seed)?;
    let mut sel_vals = vec![];
    let mut structured_features = vec![];
    for i in 0 .. ctx.tree_count {
        let mut vals = vec![];
        let mut feats = vec![];
        for j in 0 .. ctx.feature_count {
            let feature = features[i * ctx.feature_count + j];
            let ratio = ratios[i * ctx.feature_count + j];
            vals.push(truncate(&(ranges[feature] * ratio + mins[feature]), ctx.decimal_precision)?);
            feats.push(feature);
        }
        sel_vals.push(vals);
        structured_features.push(feats);
    }
    let mut disc_subsets = vec![];
    for i in 0 .. ctx.tree_count {
        let mut disc_set = vec![];
        for j in 0 .. ctx.feature_count {
            let val = sel_vals[i][j];
            let feat = structured_features[i][j];
            let col = data[feat].iter().map(|x| (*x >= val) as usize).collect::<Vec<usize>>();
            disc_set.push(col);
        }
        disc_subsets.push(disc_set);
    }
    Ok((disc_subsets, structured_features, sel_vals))
}