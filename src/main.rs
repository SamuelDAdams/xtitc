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
    pub emulate_fpp: bool,
    pub discretize_per_node: bool,
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
    let (ctx, data, classes, data_test, classes_test) = init(&fileloc.to_string()).unwrap();
    if ctx.discretize_per_node {

        let (disc_data, feature_selectors, feature_values) = xt_preprocess_per_node(&data, &ctx).unwrap();
        let trees = sid3t_per_node(&disc_data, &classes, &feature_selectors, &feature_values, &ctx).unwrap();
        let argmax_acc = classify_argmax(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        let softvote_acc = classify_softvote(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        println!("argmax acc = {}, softvote_acc = {}", argmax_acc * 100.0, softvote_acc * 100.0);

    } else {

        let (disc_data, feature_selectors, feature_values) = xt_preprocess(&data, &ctx).unwrap();
        let trees = sid3t(&disc_data, &classes, &feature_selectors, &feature_values, &ctx).unwrap();
        let argmax_acc = classify_argmax(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        let softvote_acc = classify_softvote(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        println!("argmax acc = {}, softvote_acc = {}", argmax_acc * 100.0, softvote_acc * 100.0);

    }
}

pub fn sid3t(data: &Vec<Vec<Vec<usize>>>, classes: &Vec<Vec<usize>>, subset_indices: &Vec<Vec<usize>>, split_points: &Vec<Vec<f64>>, ctx: &Context) -> Result<Vec<Vec<Node>>, Box<dyn Error>>{
    let max_depth = ctx.max_depth;
    let epsilon = ctx.epsilon;
    let tree_count = ctx.tree_count;
    let instance_count = ctx.instance_count;
    let class_label_count = 2;

    let mut trees = vec![vec![Node {
        attribute: 0,
        value: 0f64,
        frequencies: vec![],
    }]; tree_count];

    let mut transaction_subsets = vec![vec![vec![1usize; instance_count]]; ctx.tree_count]; //3d treecount x nodes_to_process_per_tree x instance_count
    let mut ances_class_bits = vec![vec![0usize];tree_count];
    for d in 0 .. max_depth {
        let nodes_to_process_per_tree = 2usize.pow(d as u32);
        let is_max_depth = d == max_depth - 1; // Are we at the final layer?
        // let number_of_nodes_to_process = nodes_to_process_per_tree * tree_count;
        //find frequencies
        let mut freqs = vec![vec![vec![]; nodes_to_process_per_tree]; tree_count];
        let mut counts = vec![vec![0usize; nodes_to_process_per_tree]; tree_count];
        let mut transaction_subsets_by_class = vec![vec![vec![]; nodes_to_process_per_tree]; tree_count];
        let mut this_layer_class_bits = vec![vec![0usize; nodes_to_process_per_tree]; tree_count];
        for t in 0 .. tree_count {
            for n in 0 .. nodes_to_process_per_tree {
                for b in 0 .. class_label_count {
                    let transaction_subset: Vec<usize> = transaction_subsets[t][n].iter().zip(classes[b].iter()).map(|(x, y)| *x * *y).collect();
                    let freq: usize = transaction_subset.iter().sum();
                    if freq == 0 {
                        this_layer_class_bits[t][n] = 1; //constant class 
                    }
                    transaction_subsets_by_class[t][n].push(transaction_subset);
                    freqs[t][n].push(freq);
                }
                counts[t][n] = freqs[t][n].iter().sum();
                if (counts[t][n] as f64) < (instance_count as f64 * epsilon) {
                    this_layer_class_bits[t][n] = 1; //less than epsilon
                }
            }
        }

        println!("{:?}", freqs);

        //if last layer, create nodes and return
        if is_max_depth {
            for t in 0 .. tree_count {
                for n in 0 .. nodes_to_process_per_tree {
                    let val = if ances_class_bits[t][n] == 1 {vec![0;class_label_count]} else {freqs[t][n].clone()};
                    trees[t].push(Node {
                        attribute: 0,
                        value: 0.,
                        frequencies: val,
                    });
                }
            }
            return Ok(trees);
        }
        //if is constant or is below threshold of epsilon*instance_count and a parent node has not classified, have the node classify with the frequencies
        let mut next_layer_tbvs = vec![vec![]; tree_count];
        let mut next_layer_class_bits = vec![vec![]; tree_count];
        let gini_argmax = gini_impurity(&data, nodes_to_process_per_tree, &classes, &transaction_subsets.clone().into_iter().flatten().collect(), &ctx)?;
        println!("{:?}", gini_argmax);
        for t in 0 .. tree_count {
            for n in 0 .. nodes_to_process_per_tree {
                let index= gini_argmax[t * nodes_to_process_per_tree + n];
                let split = split_points[t][index];
                let feat_selected = subset_indices[t][index];

                let right_tbv: Vec<usize> = transaction_subsets[t][n].iter().zip(&data[t][index]).map(|(x, y)| *x & *y).collect();
                let left_tbv: Vec<usize> = transaction_subsets[t][n].iter().zip(&data[t][index]).map(|(x, y)| *x & (*y ^ 1)).collect();
                let frequencies = if ances_class_bits[t][n] == 0 && this_layer_class_bits[t][n] == 1 {freqs[t][n].clone()} else {vec![0; class_label_count]};
                println!("Tree {:?} Node {:?} Left TBV size: {:?} Right TBV size: {:?}", t, n, left_tbv.iter().sum::<usize>(), right_tbv.iter().sum::<usize>());
                
                next_layer_tbvs[t].push(left_tbv);
                next_layer_tbvs[t].push(right_tbv);
                next_layer_class_bits[t].push(ances_class_bits[t][n] | this_layer_class_bits[t][n]);
                next_layer_class_bits[t].push(ances_class_bits[t][n] | this_layer_class_bits[t][n]);

                trees[t].push(Node {
                    attribute: feat_selected,
                    value: split,
                    frequencies: frequencies
                })

            }
        }
        transaction_subsets = next_layer_tbvs;
        ances_class_bits = next_layer_class_bits;

        //find the gini argmax, use that value as the split point

        //create the new transaction subsets
        
    }

    Ok(trees)
}

pub fn sid3t_per_node(data: &Vec<Vec<Vec<Vec<usize>>>>, classes: &Vec<Vec<usize>>, subset_indices: &Vec<Vec<Vec<usize>>>, split_points: &Vec<Vec<Vec<f64>>>, ctx: &Context) -> Result<Vec<Vec<Node>>, Box<dyn Error>>{
    let max_depth = ctx.max_depth;
    let epsilon = ctx.epsilon;
    let tree_count = ctx.tree_count;
    let instance_count = ctx.instance_count;
    let class_label_count = 2;

    let mut trees = vec![vec![Node {
        attribute: 0,
        value: 0f64,
        frequencies: vec![],
    }]; tree_count];

    let mut transaction_subsets = vec![vec![vec![1usize; instance_count]]; ctx.tree_count]; //3d treecount x nodes_to_process_per_tree x instance_count
    let mut ances_class_bits = vec![vec![0usize];tree_count];
    let mut nodes_processed_thus_far: usize = 0;
    for d in 0 .. max_depth {
        let nodes_to_process_per_tree = 2usize.pow(d as u32);
        let is_max_depth = d == max_depth - 1; // Are we at the final layer?
        // let number_of_nodes_to_process = nodes_to_process_per_tree * tree_count;
        //find frequencies
        let mut freqs = vec![vec![vec![]; nodes_to_process_per_tree]; tree_count];
        let mut counts = vec![vec![0usize; nodes_to_process_per_tree]; tree_count];
        let mut transaction_subsets_by_class = vec![vec![vec![]; nodes_to_process_per_tree]; tree_count];
        let mut this_layer_class_bits = vec![vec![0usize; nodes_to_process_per_tree]; tree_count];
        for t in 0 .. tree_count {
            for n in 0 .. nodes_to_process_per_tree {
                for b in 0 .. class_label_count {
                    let transaction_subset: Vec<usize> = transaction_subsets[t][n].iter().zip(classes[b].iter()).map(|(x, y)| *x * *y).collect();
                    let freq: usize = transaction_subset.iter().sum();
                    if freq == 0 {
                        this_layer_class_bits[t][n] = 1; //constant class 
                    }
                    transaction_subsets_by_class[t][n].push(transaction_subset);
                    freqs[t][n].push(freq);
                }
                counts[t][n] = freqs[t][n].iter().sum();
                if (counts[t][n] as f64) < (instance_count as f64 * epsilon) {
                    this_layer_class_bits[t][n] = 1; //less than epsilon
                }
            }
        }

        println!("{:?}", freqs);

        //if last layer, create nodes and return
        if is_max_depth {
            for t in 0 .. tree_count {
                for n in 0 .. nodes_to_process_per_tree {
                    let val = if ances_class_bits[t][n] == 1 {vec![0;class_label_count]} else {freqs[t][n].clone()};
                    trees[t].push(Node {
                        attribute: 0,
                        value: 0.,
                        frequencies: val,
                    });
                }
            }
            return Ok(trees);
        }
        //if is constant or is below threshold of epsilon*instance_count and a parent node has not classified, have the node classify with the frequencies
        let mut next_layer_tbvs = vec![vec![]; tree_count];
        let mut next_layer_class_bits = vec![vec![]; tree_count];
        let mut disc_layer_data = vec![];
        for t in 0 .. tree_count {
            for n in nodes_processed_thus_far/tree_count .. nodes_processed_thus_far/tree_count + nodes_to_process_per_tree {
                disc_layer_data.push(data[t][n].clone());
            }
        }
        let gini_argmax = gini_impurity(&disc_layer_data, 1, &classes, &transaction_subsets.clone().into_iter().flatten().collect(), &ctx)?;
        println!("{:?}", gini_argmax);
        for t in 0 .. tree_count {
            for n in 0 .. nodes_to_process_per_tree {
                let index= gini_argmax[t * nodes_to_process_per_tree + n];
                let split = split_points[t][nodes_processed_thus_far/tree_count + n][index];
                let feat_selected = subset_indices[t][nodes_processed_thus_far/tree_count + n][index];

                let right_tbv: Vec<usize> = transaction_subsets[t][n].iter().zip(&data[t][nodes_processed_thus_far/tree_count + n][index]).map(|(x, y)| *x & *y).collect();
                let left_tbv: Vec<usize> = transaction_subsets[t][n].iter().zip(&data[t][nodes_processed_thus_far/tree_count + n][index]).map(|(x, y)| *x & (*y ^ 1)).collect();
                let frequencies = if ances_class_bits[t][n] == 0 && this_layer_class_bits[t][n] == 1 {freqs[t][n].clone()} else {vec![0; class_label_count]};
                println!("Tree {:?} Node {:?} Left TBV size: {:?} Right TBV size: {:?}", t, n, left_tbv.iter().sum::<usize>(), right_tbv.iter().sum::<usize>());
                
                next_layer_tbvs[t].push(left_tbv);
                next_layer_tbvs[t].push(right_tbv);
                next_layer_class_bits[t].push(ances_class_bits[t][n] | this_layer_class_bits[t][n]);
                next_layer_class_bits[t].push(ances_class_bits[t][n] | this_layer_class_bits[t][n]);

                trees[t].push(Node {
                    attribute: feat_selected,
                    value: split,
                    frequencies: frequencies
                })

            }
        }
        transaction_subsets = next_layer_tbvs;
        ances_class_bits = next_layer_class_bits;

        //find the gini argmax, use that value as the split point

        //create the new transaction subsets
        nodes_processed_thus_far += nodes_to_process_per_tree * tree_count;
    }

    Ok(trees)
}

pub fn init(cfg_file: &String) -> Result<(Context, Vec<Vec<f64>>, Vec<Vec<usize>>, Vec<Vec<f64>>, Vec<Vec<usize>>), Box<dyn Error>> {
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
    let epsilon: f64 = settings.get_float("epsilon")? as f64;
    let emulate_fpp: bool = settings.get_bool("emulate_fpp")? as bool;
    let discretize_per_node: bool = settings.get_bool("discretize_per_node")? as bool;
    let decimal_precision: usize = settings.get_int("decimal_precision")? as usize;
    let original_attr_count = attribute_count;
    let bin_count = 2usize;

    let data = matrix_csv_to_float_vec(&settings.get_str("data")?)?;
    let data = data.iter().map(|x| x.iter().map(|y| truncate(y, decimal_precision, emulate_fpp).unwrap()).collect()).collect();
    let data = transpose(&data)?;
    let mut classes = matrix_csv_to_float_vec(&settings.get_str("classes")?)?;

    classes = transpose(&classes)?;
    let classes2d: Vec<Vec<usize>> = classes.iter().map(|x| x.iter().map(|y| *y as usize).collect()).collect();

    let data_test = matrix_csv_to_float_vec(&settings.get_str("data_test")?)?;
    let data_test = data_test.iter().map(|x| x.iter().map(|y| truncate(y, decimal_precision, emulate_fpp).unwrap()).collect()).collect();
    let mut classes_test = matrix_csv_to_float_vec(&settings.get_str("classes_test")?)?;

    classes_test = transpose(&classes_test)?;
    let classes_test2d: Vec<Vec<usize>> = classes_test.iter().map(|x| x.iter().map(|y| *y as usize).collect()).collect();

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
        emulate_fpp,
        discretize_per_node,
    };

    Ok((c, data, classes2d, data_test, classes_test2d))
}

pub fn xt_preprocess(data: &Vec<Vec<f64>>, ctx: &Context) -> Result<(Vec<Vec<Vec<usize>>>, Vec<Vec<usize>>, Vec<Vec<f64>>), Box<dyn Error>>{
    let maxes: Vec<f64> = data.iter().map(|x| x.iter().cloned().fold(0./0., f64::max)).collect();
    let mins: Vec<f64> = data.iter().map(|x| x.iter().cloned().fold(1./0., f64::min)).collect();
    let ratios = get_ratios(ctx.feature_count * ctx.tree_count, ctx.decimal_precision, ctx.seed, ctx.emulate_fpp)?;
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
            vals.push(truncate(&(ranges[feature] * ratio + mins[feature]), ctx.decimal_precision, ctx.emulate_fpp)?);
            feats.push(feature);
        }
        sel_vals.push(vals);
        structured_features.push(feats);
    }


    for val in  structured_features.clone() {
        for d in val {
            println!("{}", d);
        }
    }

    for val in  sel_vals.clone() {
        for d in val {
            println!("{}", d);
        }
    }

    // let structured_features = vec![vec![26,11,24,25,3], vec![27,6,16,16,11],vec![21,21,15,25,1],vec![10,2,3,24,17],vec![9,22,1,22,4]];
    // let sel_vals = 
    // vec![vec![396.8666679676933,3582.2440809212403,214.3479512467384,281.3924332730062,820604.1810913497], 
    // vec![175.8725853883068,190.8717849781846,127.57492025363022,107.81530768751179,2095.11544598215],
    // vec![33470.962593262535,146.93060475706866,127.48053427465092,835.5306079698639,11554.328419675272],
    // vec![394.5404450376706,47687.021836217966,759219.9524914473,94.10921783948281,5.881694314585763],
    // vec![74.70761933260795,100868.49415164371,20241.49782356876,644.5577668168057,152.90517372090724]];

    let mut disc_subsets = vec![];
    for i in 0 .. ctx.tree_count {
        let mut disc_set = vec![];
        for j in 0 .. ctx.feature_count {
            let val = sel_vals[i][j];
            let feat = structured_features[i][j];
            let col = data[feat].iter().map(|x| if *x >= val {1} else {0}).collect::<Vec<usize>>();
            disc_set.push(col);
        }
        disc_subsets.push(disc_set);
    }
    Ok((disc_subsets, structured_features, sel_vals))
}

pub fn xt_preprocess_per_node(data: &Vec<Vec<f64>>, ctx: &Context) -> Result<(Vec<Vec<Vec<Vec<usize>>>>, Vec<Vec<Vec<usize>>>, Vec<Vec<Vec<f64>>>), Box<dyn Error>>{
    let nodes_per_tree = 2usize.pow(ctx.max_depth as u32) - 1;
    let maxes: Vec<f64> = data.iter().map(|x| x.iter().cloned().fold(0./0., f64::max)).collect();
    let mins: Vec<f64> = data.iter().map(|x| x.iter().cloned().fold(1./0., f64::min)).collect();
    let ratios = get_ratios(ctx.feature_count * ctx.tree_count * nodes_per_tree, ctx.decimal_precision, ctx.seed, ctx.emulate_fpp)?;
    let ranges: Vec<f64> = maxes.iter().zip(mins.iter()).map(|(max , min)| max - min).collect();
    let features = get_features(ctx.feature_count * ctx.tree_count * nodes_per_tree, ctx.attribute_count, ctx.seed)?;
    let mut sel_vals = vec![];
    let mut structured_features = vec![];
    for i in 0 .. ctx.tree_count {
        let mut tree_vals = vec![];
        let mut tree_feats = vec![];
        for j in 0 .. nodes_per_tree {
            let mut vals = vec![];
            let mut feats = vec![];
            for k in 0 .. ctx.feature_count {
                let feature = features[i * ctx.feature_count * nodes_per_tree + j * ctx.feature_count + k];
                let ratio = ratios[i * ctx.feature_count * nodes_per_tree + j * ctx.feature_count + k];
                vals.push(truncate(&(ranges[feature] * ratio + mins[feature]), ctx.decimal_precision, ctx.emulate_fpp)?);
                feats.push(feature);
            } 
            tree_feats.push(feats);
            tree_vals.push(vals);
        }
        sel_vals.push(tree_vals);
        structured_features.push(tree_feats); 
    }

    // let structured_features = vec![vec![26,11,24,25,3], vec![27,6,16,16,11],vec![21,21,15,25,1],vec![10,2,3,24,17],vec![9,22,1,22,4]];
    // let sel_vals = 
    // vec![vec![396.8666679676933,3582.2440809212403,214.3479512467384,281.3924332730062,820604.1810913497], 
    // vec![175.8725853883068,190.8717849781846,127.57492025363022,107.81530768751179,2095.11544598215],
    // vec![33470.962593262535,146.93060475706866,127.48053427465092,835.5306079698639,11554.328419675272],
    // vec![394.5404450376706,47687.021836217966,759219.9524914473,94.10921783948281,5.881694314585763],
    // vec![74.70761933260795,100868.49415164371,20241.49782356876,644.5577668168057,152.90517372090724]];

    let mut disc_subsets = vec![];
    for i in 0 .. ctx.tree_count {
        let mut tree_disc_set = vec![];
        for j in 0 .. nodes_per_tree {
            let mut disc_set = vec![];
            for k in 0 .. ctx.feature_count {
                let val = sel_vals[i][j][k];
                let feat = structured_features[i][j][k];
                let col = data[feat].iter().map(|x| if *x >= val {1} else {0}).collect::<Vec<usize>>();
                disc_set.push(col);
            }
            tree_disc_set.push(disc_set);
        }
        disc_subsets.push(tree_disc_set);
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


pub fn gini_impurity(disc_data: &Vec<Vec<Vec<usize>>>, number_of_nodes_per_tree: usize, labels: &Vec<Vec<usize>>, 
    active_rows: &Vec<Vec<usize>>, ctx: &Context) -> Result<Vec<usize>, Box<dyn Error>> {

        let length = active_rows.len();

        // let mut disc_data_ext = vec![];
        
        // for tree in disc_data {
        //     for _i in 0.. number_of_nodes_per_tree {
        //         disc_data_ext.push(tree.clone());
        //     }
        // }

        let mut gini_index_per_tree = vec![0; length];
        println!("{}", length);

        // assumes binary classification
        let lab_row_wise = labels[1].clone();

        for t in 0.. length {

            // row wise data
            let row_wise = transpose(&disc_data[t / number_of_nodes_per_tree]).unwrap();

            let mut active_data = vec![];
            let mut active_labels = vec![];
            // if row is valid, append to temp
            for r in 0.. ctx.instance_count {
                // println!("{}",active_rows[t][r]);
                if active_rows[t][r] == 1 {
                    active_data.push(row_wise[r].clone());
                    active_labels.push(lab_row_wise[r])
                }
            }

            let active_data = transpose(&active_data).unwrap();
            if active_data.len() == 0 as usize {
                gini_index_per_tree[t] = 0;
                continue;
            }
            let mut gini_vals = vec![];
            // it's weird like this because we no longer OHE with bin_count in mind
            for k in 0.. ctx.feature_count {
                gini_vals.push(gini_col(&active_data[k], &active_labels, ctx).unwrap());
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
    
    pub fn gini_col(cols: &Vec<usize>, labels: &Vec<usize>, ctx: &Context) -> Result<f64, Box<dyn Error>> {

        let mut bins = vec![vec![0 as f64; ctx.class_label_count]; ctx.bin_count];
        // weirdness is artifact of needing bin_count, keeping it here to remind myself
        let row = cols;
        
        let active_instance_count = row.len();

        for r in 0.. active_instance_count {
            bins[row[r]][labels[r]] += 1 as f64;
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
            gini += ((val_0 * val_0) + (val_1 * val_1)) * (weights[j]/weight_sum);
        }


        Ok(gini)
    }


    pub fn classify_argmax(trees: &Vec<Vec<Node>>, transactions: &Vec<Vec<f64>>, labels: &Vec<usize>, ctx: &Context) 
    -> Result<f64, Box<dyn Error>> {

        // for tree in trees {
        //     for node in tree {
        //         println!("{} {} {:?}", node.attribute, node.value, node.frequencies);
        //     }
        // }

        // pub attribute: usize,
        // pub value: f64,
        // pub frequencies: Vec<usize>

        let bin_count = 2;
        
        let ensemble = trees;
        let mut correctly_classified = 0;
        let total_rows =  transactions.len();

        let depth = ctx.max_depth - 1;

        let mut transaction_index = 0;

        for transaction in transactions {

            let mut votes = vec![0; ctx.class_label_count];

            for tree in ensemble {

                let mut vote = 0;
                
                let mut current_node = 1;

                for d in 0.. depth {
                    let chosen_attr = tree[current_node].attribute;
                    let splits = [tree[current_node].value].to_vec(); // for the sake of keeping things the same

                    let val = transaction[chosen_attr];

                    let mut bin = 0;
                    for split in splits {
                        if val < split {break};
                        bin += 1;
                    }

                    current_node = bin_count * current_node + bin;

                    //println!("0: {}, 1: {}", tree[current_node].frequencies[0], tree[current_node].frequencies[1]);

                    // 'argmax'
                    if tree[current_node].frequencies[0] < tree[current_node].frequencies[1] {
                        vote = 1;
                    } else {
                        vote = 0;
                    }
                    if d + 1 == depth {
                        votes[vote] += 1;
                    }

                }
                
            }

            let mut largest_index = 0;
            let mut largest = votes[largest_index];

            for i in 1.. votes.len() {
                if largest < votes[i] {
                    largest_index = i;
                    largest = votes[i];
                }
            }

            if labels[transaction_index] as usize == largest_index {
                correctly_classified += 1;
            }

            transaction_index += 1;

        }

        Ok((correctly_classified as f64) / (total_rows as f64))
    }


    pub fn classify_softvote(trees: &Vec<Vec<Node>>, transactions: &Vec<Vec<f64>>, labels: &Vec<usize>, ctx: &Context) 
    -> Result<f64, Box<dyn Error>> {

        let bin_count = 2;
        
        let ensemble = trees;
        let mut correctly_classified = 0;
        let total_rows =  transactions.len();

        let depth = ctx.max_depth - 1;

        let mut transaction_index = 0;

        for transaction in transactions {

            let mut votes = vec![0 as f64; ctx.class_label_count];

            for tree in ensemble {

                let mut current_node = 1;

                for d in 0.. depth {
                    let chosen_attr = tree[current_node].attribute;
                    let splits = [tree[current_node].value].to_vec(); // for the sake of keeping things the same

                    let val = transaction[chosen_attr];

                    let mut bin = 0;
                    for split in splits {
                        if val < split {break};
                        bin += 1;
                    }

                    current_node = bin_count * current_node + bin;

                    if tree[current_node].frequencies[0] != 0 || tree[current_node].frequencies[1] != 0 {
                        let zero: f64 = tree[current_node].frequencies[0] as f64;
                        let one: f64 = tree[current_node].frequencies[1] as f64;
                        let total: f64 = zero + one;

                        votes[0] += zero/total;
                        votes[1] += one/total;
                    }

                }
                
            }

            let mut largest_index = 0;
            let mut largest = votes[largest_index];

            // println!("{:?}", votes);

            for i in 1.. votes.len() {
                if largest < votes[i] {
                    largest_index = i;
                    largest = votes[i];
                }
            }

            if labels[transaction_index] as usize == largest_index {
                correctly_classified += 1;
            }

            transaction_index += 1;

        }

        Ok((correctly_classified as f64) / (total_rows as f64))
    }