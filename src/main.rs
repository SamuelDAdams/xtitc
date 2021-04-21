mod util;
use std::{error::Error};
use util::*;

#[derive(Default)]
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

fn main() {
    let fileloc = "settings/settings1.toml";
    //load settings
    let (ctx, data, classes) = init(&fileloc.to_string()).unwrap();
    let (disc_data, feature_selectors, feature_values) = xt_preprocess(&data, &ctx).unwrap();
    //let 
    //preprocess dataset according to the settings
    
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


pub fn class_frequencies(labels: &Vec<Vec<usize>>, active_rows: &Vec<Vec<usize>>, ctx: &Context) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {

    let mut freq_vec = vec![vec![0; ctx.class_label_count]; ctx.tree_count];

    for t in 0.. ctx.tree_count {
        // row wise data
        let row_wise = transpose(&labels).unwrap();

        let mut active_labels = vec![];
        // if row is valid, append to temp
        for r in 0.. ctx.instance_count {
            if active_rows[t][r] == 1 {
                active_labels.push(row_wise[r].clone())
            }
        }

        let active_labels = transpose(&active_labels).unwrap();
        for i in 0.. ctx.class_label_count {
            freq_vec[t][i] = active_labels[i].iter().sum();
        }
    }

    Ok(freq_vec)

}


pub fn gini_impurity(disc_data: &Vec<Vec<Vec<usize>>>, labels: &Vec<Vec<usize>>, 
    active_rows: &Vec<Vec<usize>>, ctx: &Context) -> Result<Vec<usize>, Box<dyn Error>> {

        let mut gini_index_per_tree = vec![0; ctx.tree_count];
        // assumes binary classification
        let lab_row_wise = labels[1].clone();

        for t in 0.. ctx.tree_count {
            // row wise data
            let row_wise = transpose(&disc_data[t]).unwrap();
            

            let mut active_data = vec![];
            let mut active_labels = vec![];
            // if row is valid, append to temp
            for r in 0.. ctx.instance_count {
                if active_rows[t][r] == 1 {
                    active_data.push(row_wise[r].clone());
                    active_labels.push(lab_row_wise[r])
                }
            }

            let active_data = transpose(&active_data).unwrap();
            let mut gini_vals = vec![];
            let b = ctx.bin_count;
            for k in 0.. ctx.feature_count {
                gini_vals.push(gini_col(&active_data[k * b.. (k + 1) * b].to_vec(), &active_labels, ctx).unwrap());
            }
            gini_index_per_tree[t] = argmax(&gini_vals).unwrap();
        }

        Ok(gini_index_per_tree)
    }

    pub fn argmax(v: &Vec<f64>) -> Result<usize, Box<dyn Error>> {
        let mut max_val = v[0];
        let mut max_index = 0;

        for i in 0.. v.len() {
            let val = v[i];
            if val > max_val {
                max_val = val;
                max_index = i
            }
        }
        Ok(max_index)
    }
    
    pub fn gini_col(cols: &Vec<Vec<usize>>, labels: &Vec<usize>, ctx: &Context) -> Result<f64, Box<dyn Error>> {

        let mut bins = vec![vec![0 as f64; ctx.class_label_count]; ctx.bin_count];
        let rows = transpose(cols).unwrap();
        
        let active_instance_count = rows.len();

        for r in 0.. active_instance_count {
            let row = &rows[r];
            for j in 0.. ctx.bin_count {
                if row[j] == 1 {
                    bins[j][labels[r]] += 1 as f64;
                }
            }
        }

        let mut weights = vec![];
        for row in bins.clone() {
            weights.push(row.iter().sum());
        }
        let weight_sum: f64 = weights.iter().sum();

        let mut gini = 0 as f64;
        // Assumes binary classificaiton
        for j in 0.. ctx.bin_count {
            let val_0: f64 = bins[j][0]/weights[j];
            let val_1: f64 = bins[j][1]/weights[j];
            gini += ((val_0 * val_0) + (val_1 * val_1))/(weights[j]/weight_sum);
        }

        Ok(gini)
    }