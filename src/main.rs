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
        let trees = sid3t_per_node(&disc_data, &classes, &feature_selectors, &feature_values, &ctx).unwrap();
        let argmax_acc = classify_argmax(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        let softvote_acc = classify_softvote(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        println!("argmax acc = {}, softvote_acc = {}", argmax_acc * 100.0, softvote_acc * 100.0);

    } else {

        let (disc_data, feature_selectors, feature_values) = xt_preprocess(&data, &ctx).unwrap();
        for t in 0.. disc_data.len() {
            for k in 0.. disc_data[t].len() {
                let mut z = 0;
                let mut o = 0;

                for val in disc_data[t][k].clone() {
                    if val == 0 {
                        z += 1
                    } else {
                        o += 1
                    }
                }
                // println!("For tree {} and feature {}, we have {} < split, {} >= split", t, k, z, o)
            }
        }

        let mode: &str = "traditional";
        let trees = sid3t(&disc_data.clone(), &classes, &feature_selectors, &feature_values, &ctx, mode).unwrap();
        let argmax_acc = classify_argmax(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        let softvote_acc = classify_softvote(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        println!("argmax acc = {}, softvote_acc = {}", argmax_acc * 100.0, softvote_acc * 100.0);
        let mode: &str = "secure_old";
        let trees = sid3t(&disc_data.clone(), &classes, &feature_selectors, &feature_values, &ctx, mode).unwrap();
        let argmax_acc = classify_argmax(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        let softvote_acc = classify_softvote(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        println!("argmax acc = {}, softvote_acc = {}", argmax_acc * 100.0, softvote_acc * 100.0);
        let mode: &str = "secure_new";
        let trees = sid3t(&disc_data.clone(), &classes, &feature_selectors, &feature_values, &ctx, mode).unwrap();
        let argmax_acc = classify_argmax(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        let softvote_acc = classify_softvote(&trees.clone(), &data_test.clone(), &classes_test[1].clone(), &ctx).unwrap();
        println!("argmax acc = {}, softvote_acc = {}", argmax_acc * 100.0, softvote_acc * 100.0);

    }
}

pub fn sid3t(data: &Vec<Vec<Vec<usize>>>, classes: &Vec<Vec<usize>>, subset_indices: &Vec<Vec<usize>>, split_points: &Vec<Vec<f64>>, ctx: &Context, gini: &str) -> Result<Vec<Vec<Node>>, Box<dyn Error>>{
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

        //println!("{:?}", freqs);

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
        let mut gini_argmax: Vec<usize> = vec![];
        if gini.eq("traditional") {
            gini_argmax = gini_impurity(&data.clone(), nodes_to_process_per_tree, &classes, &transaction_subsets.clone().into_iter().flatten().collect(), &ctx)?;
        } else if gini.eq("secure_old") {
            gini_argmax = gini_impurity_secure_algorithm_old(&data.clone(), d, &classes, &transaction_subsets.clone().into_iter().flatten().collect(), &ctx)?;
        } else if gini.eq("secure_new") {
            gini_argmax = gini_impurity_secure_algorithm_updated(&data.clone(), d, &classes, &transaction_subsets.clone().into_iter().flatten().collect(), &ctx)?;
        }
        
        println!("{:?}", gini_argmax);
        for t in 0 .. tree_count {
            for n in 0 .. nodes_to_process_per_tree {
                let index = gini_argmax[t * nodes_to_process_per_tree + n];
                let split = split_points[t][index];
                let feat_selected = subset_indices[t][index];

                let right_tbv: Vec<usize> = transaction_subsets[t][n].iter().zip(&data[t][index]).map(|(x, y)| *x & *y).collect();
                let left_tbv: Vec<usize> = transaction_subsets[t][n].iter().zip(&data[t][index]).map(|(x, y)| *x & (*y ^ 1)).collect();
                let frequencies = if ances_class_bits[t][n] == 0 && this_layer_class_bits[t][n] == 1 {freqs[t][n].clone()} else {vec![0; class_label_count]};
                //println!("Tree {:?} Node {:?} Left TBV size: {:?} Right TBV size: {:?}", t, n, left_tbv.iter().sum::<usize>(), right_tbv.iter().sum::<usize>());
                
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

        println!("{}", d);

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

        //println!("{:?}", freqs);

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
        //println!("{:?}", gini_argmax);
        for t in 0 .. tree_count {
            for n in 0 .. nodes_to_process_per_tree {
                let index= gini_argmax[t * nodes_to_process_per_tree + n];
                let split = split_points[t][nodes_processed_thus_far/tree_count + n][index];
                let feat_selected = subset_indices[t][nodes_processed_thus_far/tree_count + n][index];

                let right_tbv: Vec<usize> = transaction_subsets[t][n].iter().zip(&data[t][nodes_processed_thus_far/tree_count + n][index]).map(|(x, y)| *x & *y).collect();
                let left_tbv: Vec<usize> = transaction_subsets[t][n].iter().zip(&data[t][nodes_processed_thus_far/tree_count + n][index]).map(|(x, y)| *x & (*y ^ 1)).collect();
                let frequencies = if ances_class_bits[t][n] == 0 && this_layer_class_bits[t][n] == 1 {freqs[t][n].clone()} else {vec![0; class_label_count]};
                //println!("Tree {:?} Node {:?} Left TBV size: {:?} Right TBV size: {:?}", t, n, left_tbv.iter().sum::<usize>(), right_tbv.iter().sum::<usize>());
                
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
        println!("vec!{:?}", val);
    }

    for val in  sel_vals.clone() {
        println!("vec!{:?}", val);
    }

    

    let structured_features = vec![
        vec![15, 14, 0, 19, 29],
        vec![19, 25, 28, 17, 14],
        vec![27, 0, 15, 3, 14],
        vec![7, 11, 6, 27, 8],
        vec![0, 26, 2, 16, 24],
        vec![21, 6, 6, 24, 13],
        vec![4, 24, 27, 12, 9],
        vec![22, 16, 3, 4, 14],
        vec![13, 11, 28, 24, 25],
        vec![26, 23, 16, 26, 20],
        vec![26, 1, 6, 26, 12],
        vec![4, 15, 28, 8, 14],
        vec![21, 21, 0, 22, 9],
        vec![20, 17, 10, 28, 2],
        vec![3, 1, 5, 14, 12],
        vec![0, 27, 22, 18, 1],
        vec![24, 27, 28, 19, 7],
        vec![4, 21, 7, 13, 11],
        vec![12, 7, 19, 0, 3],
        vec![8, 23, 19, 16, 26]  
    ];

    let sel_vals = 
    vec![
        vec![107.96484375, 26.130859375, 22555.54296875, 3.341796875, 83.4775390625],
        vec![19.931640625, 82.650390625, 315.03125, 20.8095703125, 22.68359375],
        vec![84.9697265625, 7020.529296875, 125.388671875, 1363377.44140625, 22.626953125],
        vec![60.341796875, 452.994140625, 285.6552734375, 241.8369140625, 192.4248046875],
        vec![8878.40625, 59.4140625, 146982.2265625, 24.8837890625, 124.111328125],
        vec![22222.861328125, 89.943359375, 218.197265625, 99.267578125, 122878.177734375],
        vec![72.5341796875, 93.3515625, 194.0947265625, 10727.576171875, 79.048828125],
        vec![120638.30078125, 94.138671875, 2046344.23828125, 85.298828125, 4.98828125],
        vec![15077.998046875, 2989.3564453125, 192.169921875, 180.8974609375, 1022.7705078125],
        vec![1042.03125, 923973.828125, 108.2294921875, 1149.43359375, 33080.78125],
        vec![715.25390625, 11546.923828125, 413.908203125, 484.453125, 5671.7890625],
        vec![85.5146484375, 31.248046875, 246.6650390625, 281.6845703125, 16.1337890625],
        vec![27650.107421875, 29821.005859375, 21606.83984375, 67879.7265625, 65.681640625],
        vec![19299.53125, 37.0263671875, 600.6435546875, 631.1025390625, 76746.6796875],
        vec![1116521.97265625, 25641.533203125, 61.724609375, 12.5146484375, 10506.0078125],
        vec![18167.791015625, 269.6865234375, 160469.27734375, 35.2265625, 12370.634765625],
        vec![164.3349609375, 134.4169921875, 660.8271484375, 17.3662109375, 188.498046875],
        vec![122.5107421875, 45234.384765625, 66.693359375, 68129.267578125, 1398.6064453125],
        vec![7242.91015625, 178.0361328125, 3.1708984375, 22278.837890625, 384184.08203125],
        vec![276.9296875, 3187855.859375, 7.4892578125, 20.6865234375, 543.8671875]
    ];

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
    // vec![
    // vec![396.8666679676933,3582.2440809212403,214.3479512467384,281.3924332730062,820604.1810913497], 
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

pub fn gini_impurity_secure_algorithm_old(disc_data: &Vec<Vec<Vec<usize>>>, depth: usize, labels: &Vec<Vec<usize>>, 
    active_rows: &Vec<Vec<usize>>, ctx: &Context) -> Result<Vec<usize>, Box<dyn Error>> {

        let base: usize = 2;
        let number_of_nodes_to_process = ctx.tree_count * base.pow((depth) as u32);

        let class_label_count = ctx.class_label_count;
        let decimal_precision = 10;
        let bin_count = ctx.bin_count;
        let feat_count = ctx.feature_count;
        let attribute_count = feat_count * bin_count;

        let alpha = 1; // Need this from ctx

        let data_instance_count = ctx.instance_count;

        let mut gini_arg_max: Vec<Vec<usize>> = vec![];

        let mut x_partitioned =
            vec![vec![vec![vec![0; bin_count]; class_label_count]; feat_count];number_of_nodes_to_process];
        let mut x2 =
            vec![vec![vec![vec![0; bin_count]; class_label_count]; feat_count];number_of_nodes_to_process];

        let mut y_partitioned =
            vec![vec![vec![0; bin_count]; feat_count]; number_of_nodes_to_process];

        let mut gini_numerators = vec![0; feat_count * number_of_nodes_to_process];


        // create u_decimal
        let mut u_decimal = vec![];
        
        assert_eq!(active_rows.len(), number_of_nodes_to_process);

        for n in 0.. number_of_nodes_to_process {
            for i in 0.. class_label_count {
                u_decimal.append(&mut protocol_mult(&active_rows[n].clone(), &labels[i].clone()))
            }
        }
        // done creating u_decimal

        // create input
        let mut input: Vec<Vec<Vec<usize>>> = vec![];

        assert_eq!(disc_data.len(), ctx.tree_count);        

        for t in 0.. ctx.tree_count {
            let tree_data = disc_data[t].clone();
            let mut ohe_disc_data_t = vec![];

            for col in tree_data {
                let mut big_col = vec![vec![]; bin_count];
                for val in col {
                    // since val is gonna be in [0, bin_count), index by it and push 1
                    big_col[val].push(1);
                    // push a 0 in all cols where val does not belong
                    for j in 0.. bin_count {
                        if val != j {
                            big_col[j].push(0)
                        }
                    }
                }
                ohe_disc_data_t.append(&mut big_col);
            }

            assert_eq!(ohe_disc_data_t.len(), attribute_count);

            for _d in 0.. 2usize.pow(depth as u32) {
                input.push(ohe_disc_data_t.clone())
            }
        }
        assert_eq! (input.len(), number_of_nodes_to_process);
        assert_eq!(input[0].len(), attribute_count);
        assert_eq!(input[0][0].len(), ctx.instance_count);
        // done creating input

        // Determine the number of transactions that are:
        // 1. in the current subset
        // 2. predict the i-th class value
        // 3. and have the j-th value of the k-th attribute for each node n


        // NOTE: Work on more meaningful names...
        let mut u_decimal_vectors = vec![];

        let mut u_decimal_extended = vec![];

        let mut discretized_sets_vectors: Vec<usize> = vec![];

        let mut discretized_sets = vec![vec![]; bin_count];


        // make vectors of active rows and discretized sets parrallel to prepare for
        // batch multiplication in order to find frequencies of classes
        for n in 0.. number_of_nodes_to_process {
            for k in 0.. feat_count {
                for i in 0.. class_label_count {

                    let mut u_decimal_clone = u_decimal[(n * class_label_count + i) * data_instance_count.. 
                        (n * class_label_count + i + 1) * data_instance_count].to_vec();

                    u_decimal_extended.append(&mut u_decimal_clone);


                    for j in 0.. bin_count {
                        // grabs column of data, right?
                        discretized_sets[j].append(&mut input[n][k * bin_count + j].clone());
                    }

                }
            }
        }

        let mut u_decimal_vectors_clone = u_decimal_vectors.clone();

        for j in 0.. bin_count {
            discretized_sets_vectors.append(&mut discretized_sets[j]);
            u_decimal_vectors.append(&mut u_decimal_extended.clone());
        }

        let batched_un_summed_frequencies =
        protocol_mult(&u_decimal_vectors, &discretized_sets_vectors);

        let total_number_of_rows = ctx.instance_count;

        let number_of_xs = batched_un_summed_frequencies.len() / (bin_count * total_number_of_rows);

        for v in 0..number_of_xs {
            
            for j in 0.. bin_count {

                // Sum up "total_number_of_rows" values to obtain frequency of classification for a particular subset
                // of data split a particular way dictated by things like the random feature chosen, and its split.
                let dp_result = batched_un_summed_frequencies
                    [(v + number_of_xs * j) * total_number_of_rows.. 
                    (v + 1 + number_of_xs * j) * total_number_of_rows].to_vec().iter().sum();


                //println!("idx: {}, max size: {}", (v + 1 + number_of_xs * j) * total_number_of_rows , batched_un_summed_frequencies.len());
                
                x_partitioned[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count]
                    [v % class_label_count][j] = dp_result;

                y_partitioned[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count][j] +=
                    dp_result;

            }

        }

        let mut all_x_values = vec![];

        for n in 0..number_of_nodes_to_process {
            for k in 0..feat_count { // should also be indexed by j?
                y_partitioned[n][k][0] =
                    alpha * y_partitioned[n][k][0] + 0;
                y_partitioned[n][k][1] =
                    alpha * y_partitioned[n][k][1] + 0;

                // will be used to find x^2
                for i in 0..class_label_count {
                    all_x_values.append(&mut x_partitioned[n][k][i]);
                }
            }
        }

        let all_x_values_squared: Vec<usize> = protocol_mult(&all_x_values, &all_x_values);

        for v in 0..all_x_values_squared.len() / 2 {
            for j in 0.. bin_count {
                x2[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count]
                [v % class_label_count][j] = all_x_values_squared[bin_count * v + j];
            }
        }



        // At this point we have all of our x, x^2 and y values. Now we can start calculation gini numerators/denominators
        let mut sum_of_x2_j =  vec![vec![vec![0; bin_count]; feat_count]; number_of_nodes_to_process];

        // let mut d_exclude_j = vec![vec![vec![Wrapping(0); bin_count]; feat_count]; number_of_nodes_to_process];
        // let mut d_include_j = vec![vec![]; number_of_nodes_to_process];

        let mut d_exclude_j = vec![];
        let mut d_include_j = vec![];

        // create vector of the 0 and 1 values for j to set us up for batch multiplicaiton.
        // also, sum all of the x^2 values over i, and push these sums over i to a vector
        // to batch multiply with the y_without_j values.
        for n in 0..number_of_nodes_to_process {

            let mut y_vals_include_j = vec![vec![]; feat_count];

            for k in 0..feat_count {

                let mut y_vals_exclude_j = vec![vec![]; bin_count];

                for j in 0.. bin_count {
                    
                    // at the j'th index of the vector, we need to 
                    // push all values at indeces that are not equal to j onto it
                    // not 100% sure about this..
                    for not_j in 0.. bin_count {
                        if j != not_j {
                            y_vals_exclude_j[j].push(y_partitioned[n][k][not_j]);
                        }
                    }

                    y_vals_include_j[k].push(y_partitioned[n][k][j]);
                    

                    let mut sum_j_values = 0;
        
                    for i in 0..class_label_count {
                        sum_j_values += x2[n][k][i][j];
                    }
                    
                    sum_of_x2_j[n][k][j] = sum_j_values;
                }
                d_exclude_j.extend(y_vals_exclude_j);
                // can be far better optimized. Named 'D' after De'Hooghs variable
                // d_exclude_j[n][k] = protocol::pairwise_mult_zq(&y_vals_exclude_j, ctx).unwrap();
                // println!("exclude {:?}", protocol::open(&d_exclude_j[n][k], ctx).unwrap()); //test
                // 
                // d_exclude_j.append(y_vals_exclude_j); what we should do?
            }
            d_include_j.extend(y_vals_include_j);
            //println!("{:?}", d_include_j);
            //d_include_j[n] = protocol::pairwise_mult_zq(&y_vals_include_j, ctx).unwrap();
            // println!("include {:?}", protocol::open(&d_include_j[n], ctx).unwrap()); //test
            // d_include_j.append(y_vals_include_j); what we should do?
        }

        let mut sum_of_x2_j_flattend = vec![];

        for n in 0.. number_of_nodes_to_process {
            for k in 0.. feat_count {
                sum_of_x2_j_flattend.append(&mut sum_of_x2_j[n][k].clone());
            }
        } 

        let d_exclude_j = protocol_par(&d_exclude_j);
        let d_include_j = protocol_par(&d_include_j);

        let gini_numerators_values_flat_unsummed = protocol_mult(&d_exclude_j, &sum_of_x2_j_flattend);

        // println!("{}", d_exclude_j_flattend.len()); //test
        // println!("{}", sum_of_x2_j_flattend.len()); //test

        for v in 0.. gini_numerators_values_flat_unsummed.len() / bin_count {
            for j in 0.. bin_count {
                gini_numerators[v] += gini_numerators_values_flat_unsummed[v * bin_count + j];
            }
        }

        // create denominators
        let mut gini_denominators: Vec<usize> = d_include_j;

        // println!("{}: {:?}", gini_numerators.len(), gini_numerators); //test
        // println!("{}: {:?}", gini_denominators.len(), gini_denominators); //test


        /////////////////////////////////////////// COMPUTE ARGMAX ///////////////////////////////////////////

        let mut current_length = feat_count;

        let mut logical_partition_lengths = vec![];

        let mut new_numerators = gini_numerators.clone();
        let mut new_denominators = gini_denominators.clone();

        // this will allow us to calculate the arg_max
        let mut past_assignments = vec![];

        loop {

            let odd_length = current_length % 2 == 1;
        
            let mut forgotten_numerators = vec![];
            let mut forgotten_denominators = vec![];
    
            let mut current_numerators = vec![];
            let mut current_denominators = vec![];
    
            // if its of odd length, make current nums/dems even lengthed, and store the 'forgotten' values
            // it should be guarenteed that we have numerators/denonminators of even length
            if odd_length {
                for n in 0 .. number_of_nodes_to_process {
                    current_numerators.append(&mut new_numerators[n * (current_length).. (n + 1) * (current_length - 1) + n].to_vec());
                    current_denominators.append(&mut new_denominators[n * (current_length).. (n + 1) * (current_length - 1) + n].to_vec());
    
                    forgotten_numerators.push(new_numerators[(n + 1) * (current_length - 1) + n]);
                    forgotten_denominators.push(new_denominators[(n + 1) * (current_length - 1) + n]);
    
                }
            } else {
                current_numerators = new_numerators.clone();
                current_denominators = new_denominators.clone();
            }
    
            // if denominators were originally indexed as d_1, d_2, d_3, d_4... then they are now represented
            // as d_2, d_1, d_4, d_3... This is helpful for comparisons
            let mut current_denominators_flipped = vec![];
            for v in 0 .. current_denominators.len()/2 {
                current_denominators_flipped.push(current_denominators[2 * v + 1]);
                current_denominators_flipped.push(current_denominators[2 * v]);
            }
    
            let product = protocol_mult(&current_numerators, &current_denominators_flipped);
    
            let mut l_operands = vec![];
            let mut r_operands = vec![];
    
            // left operands should be of the form n_1d_2, n_3d_4...
            // right operands should be of the form n_2d_1, n_4d_3...
            for v in 0..product.len()/2 {
                l_operands.push(product[2 * v]);
                r_operands.push(product[2 * v + 1]);
            }
    
            // read this as "left is greater than or equal to right." value in array will be [1] if true, [0] if false.
            let l_geq_r = protocol_geq(&l_operands, &r_operands);  
    
            // read this is "left is less than right"
            let l_lt_r: Vec<usize> = l_geq_r.iter().map(|x| 1 - x).collect();
    
            // grab the original values 
            let mut values = current_numerators.clone();
            values.append(&mut current_denominators.clone());
    
            // For the next iteration
            let mut assignments = vec![];
            // For record keeping
            let mut gini_assignments = vec![];
    
            // alternate the left/right values to help to cancel out the original values
            // that lost in their comparison
            for v in 0..l_geq_r.len() {
                assignments.push(l_geq_r[v]);
                assignments.push(l_lt_r[v]);
    
                gini_assignments.push(l_geq_r[v]);
                gini_assignments.push(l_lt_r[v]);
    
                let size = current_length/2;
                if odd_length && ((v + 1) % size == 0) {
                    gini_assignments.push(1);
                }
            }
    
            logical_partition_lengths.push(current_length);
            past_assignments.push(gini_assignments); 
    
            // EXIT CONDITION
            if 1 == (current_length/2) + (current_length % 2) {break;}
    
            assignments.append(&mut assignments.clone());
    
            let comparison_results = protocol_mult(&values, &assignments);
    
            new_numerators = vec![];
            new_denominators = vec![];
    
            // re-construct new_nums and new_dems
            for v in 0.. values.len()/4 {

                new_numerators.push(comparison_results[2 * v] + comparison_results[2 * v + 1]);
                new_denominators.push(comparison_results[values.len()/2 + 2 * v] + comparison_results[values.len()/2 + 2 * v + 1]);

                if odd_length && ((v + 1) % (current_length/2) == 0) {
                    // ensures no division by 0
                    let divisor = if current_length > 1 {current_length/2} else {1};
                    new_numerators.push(forgotten_numerators[(v/divisor)]);
                    new_denominators.push(forgotten_denominators[(v/divisor)]);
                }
    
            }
    
            current_length = (current_length/2) + (current_length % 2);
        }

        // calculates flat arg_max in a tournament bracket style
        for v in (1..past_assignments.len()).rev() {

            if past_assignments[v].len() == past_assignments[v - 1].len() {
                past_assignments[v - 1] = protocol_mult(&past_assignments[v - 1], &past_assignments[v]);
                continue;
            }

            let mut extended_past_assignment_v = vec![];
            for w in 0.. past_assignments[v].len() {
                if ((w + 1) % logical_partition_lengths[v] == 0) && ((logical_partition_lengths[v - 1] % 2) == 1) {
                    extended_past_assignment_v.push(past_assignments[v][w]);
                    continue;
                }
                extended_past_assignment_v.push(past_assignments[v][w]);
                extended_past_assignment_v.push(past_assignments[v][w]);
            }
            past_assignments[v - 1] = protocol_mult(&past_assignments[v - 1], &extended_past_assignment_v);
        }

        // un-flatten arg_max
        for n in 0.. number_of_nodes_to_process {
            if feat_count == 1 { // if there is only one attr count
                gini_arg_max.push(vec![1 as usize]);
            } else {
                gini_arg_max.push(past_assignments[0][n * feat_count.. (n + 1) * feat_count].to_vec());
            }
        }

        // ADDED: un-OHE argmax
        let mut tmp = vec![];
        for k in 0.. feat_count {
            tmp.push(k);
        }

        let mut new_gini = vec![];

        for argm in gini_arg_max {
            new_gini.push(protocol_dot(&tmp.clone(), &argm))
        }
        assert_eq!(new_gini.len(), number_of_nodes_to_process);

        Ok(new_gini)
    }

pub fn gini_impurity_secure_algorithm_updated(disc_data: &Vec<Vec<Vec<usize>>>, depth: usize, labels: &Vec<Vec<usize>>, 
    active_rows: &Vec<Vec<usize>>, ctx: &Context) -> Result<Vec<usize>, Box<dyn Error>> {

        let base: usize = 2;
        let number_of_nodes_to_process = ctx.tree_count * base.pow((depth) as u32);

        let class_label_count = ctx.class_label_count;
        let decimal_precision = 10;
        let bin_count = ctx.bin_count;
        let feat_count = ctx.feature_count;
        let attribute_count = feat_count * bin_count;

        let alpha = 1; // Need this from ctx

        let data_instance_count = ctx.instance_count;

        let mut gini_arg_max: Vec<Vec<usize>> = vec![];

        let mut x_partitioned =
            vec![vec![vec![vec![0; bin_count]; class_label_count]; feat_count];number_of_nodes_to_process];
        let mut x2 =
            vec![vec![vec![vec![0; bin_count]; class_label_count]; feat_count];number_of_nodes_to_process];

        let mut y_partitioned =
            vec![vec![vec![0; bin_count]; feat_count]; number_of_nodes_to_process];

        let mut gini_numerators = vec![0; feat_count * number_of_nodes_to_process];


        // create u_decimal
        let mut u_decimal = vec![];
        
        assert_eq!(active_rows.len(), number_of_nodes_to_process);

        for n in 0.. number_of_nodes_to_process {
            for i in 0.. class_label_count {
                u_decimal.append(&mut protocol_mult(&active_rows[n].clone(), &labels[i].clone()))
            }
        }
        // done creating u_decimal

        // create input
        let mut input: Vec<Vec<Vec<usize>>> = vec![];

        assert_eq!(disc_data.len(), ctx.tree_count);        

        for t in 0.. ctx.tree_count {
            let tree_data = disc_data[t].clone();
            let mut ohe_disc_data_t = vec![];

            for col in tree_data {
                let mut big_col = vec![vec![]; bin_count];
                for val in col {
                    // since val is gonna be in [0, bin_count), index by it and push 1
                    big_col[val].push(1);
                    // push a 0 in all cols where val does not belong
                    for j in 0.. bin_count {
                        if val != j {
                            big_col[j].push(0)
                        }
                    }
                }
                ohe_disc_data_t.append(&mut big_col);
            }

            assert_eq!(ohe_disc_data_t.len(), attribute_count);

            for _d in 0.. 2usize.pow(depth as u32) {
                input.push(ohe_disc_data_t.clone())
            }
        }
        assert_eq! (input.len(), number_of_nodes_to_process);
        assert_eq!(input[0].len(), attribute_count);
        assert_eq!(input[0][0].len(), ctx.instance_count);
        // done creating input

        // Determine the number of transactions that are:
        // 1. in the current subset
        // 2. predict the i-th class value
        // 3. and have the j-th value of the k-th attribute for each node n


        // NOTE: Work on more meaningful names...
        let mut u_decimal_vectors = vec![];

        let mut u_decimal_extended = vec![];

        let mut discretized_sets_vectors: Vec<usize> = vec![];

        let mut discretized_sets = vec![vec![]; bin_count];


        // make vectors of active rows and discretized sets parrallel to prepare for
        // batch multiplication in order to find frequencies of classes
        for n in 0.. number_of_nodes_to_process {
            for k in 0.. feat_count {
                for i in 0.. class_label_count {

                    let mut u_decimal_clone = u_decimal[(n * class_label_count + i) * data_instance_count.. 
                        (n * class_label_count + i + 1) * data_instance_count].to_vec();

                    u_decimal_extended.append(&mut u_decimal_clone);


                    for j in 0.. bin_count {
                        // grabs column of data, right?
                        discretized_sets[j].append(&mut input[n][k * bin_count + j].clone());
                    }

                }
            }
        }

        let mut u_decimal_vectors_clone = u_decimal_vectors.clone();

        for j in 0.. bin_count {
            discretized_sets_vectors.append(&mut discretized_sets[j]);
            u_decimal_vectors.append(&mut u_decimal_extended.clone());
        }

        let batched_un_summed_frequencies =
        protocol_mult(&u_decimal_vectors, &discretized_sets_vectors);

        let total_number_of_rows = ctx.instance_count;

        let number_of_xs = batched_un_summed_frequencies.len() / (bin_count * total_number_of_rows);

        for v in 0..number_of_xs {
            
            for j in 0.. bin_count {

                // Sum up "total_number_of_rows" values to obtain frequency of classification for a particular subset
                // of data split a particular way dictated by things like the random feature chosen, and its split.
                let dp_result = batched_un_summed_frequencies
                    [(v + number_of_xs * j) * total_number_of_rows.. 
                    (v + 1 + number_of_xs * j) * total_number_of_rows].to_vec().iter().sum();


                //println!("idx: {}, max size: {}", (v + 1 + number_of_xs * j) * total_number_of_rows , batched_un_summed_frequencies.len());
                
                x_partitioned[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count]
                    [v % class_label_count][j] = dp_result;

                y_partitioned[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count][j] +=
                    dp_result;

            }

        }



        // ////////////////// TEST //////////////////// RETURNS CORRECT RESULTS 


        // let mut ans = vec![0; number_of_nodes_to_process];
        // let mut g = vec![vec![0.0; feat_count]; number_of_nodes_to_process];

        // for n in 0.. number_of_nodes_to_process {
        //     for k in 0.. feat_count {
        //         let mut g_k = 0.0;
        //         for j in 0.. bin_count {
        //             let mut x2s = 0;
        //             for i in 0.. class_label_count {
        //                 let base = x_partitioned[n][k][i][j];
        //                 let x2 = base.pow(2);
        //                 x2s += x2;
        //             }
        //             g_k += x2s as f64 / y_partitioned[n][k][j] as f64;
        //         }
        //         g[n][k] = g_k;
        //     }
        //     ans[n] = argmax(&g[n].clone())?;
        // }
        // return Ok(ans);


        // ///// END TEST ////////////////////////////

        





        let mut all_x_values = vec![];

        for n in 0..number_of_nodes_to_process {
            for k in 0..feat_count { // should also be indexed by j?
                y_partitioned[n][k][0] =
                    alpha * y_partitioned[n][k][0] + 0;
                y_partitioned[n][k][1] =
                    alpha * y_partitioned[n][k][1] + 0;

                // will be used to find x^2
                for i in 0..class_label_count {
                    all_x_values.append(&mut x_partitioned[n][k][i]);
                }
            }
        }

        let all_x_values_squared: Vec<usize> = protocol_mult(&all_x_values, &all_x_values);

        for v in 0..all_x_values_squared.len() / 2 {
            for j in 0.. bin_count {
                x2[v / (feat_count * class_label_count)][(v / class_label_count) % feat_count]
                [v % class_label_count][j] = all_x_values_squared[bin_count * v + j];
            }
        }





        // ////////////////// TEST //////////////////// failed...?
        // // 
        // // returns correct result IF  we don't multiply y values b alpha and then add 1. However,
        // // still returns wrong answer if we don't do the former operations, but let the algorithm run to its end
        // // which is strange.


        // let mut ans = vec![0; number_of_nodes_to_process];
        // let mut g = vec![vec![0.0; feat_count]; number_of_nodes_to_process];

        // for n in 0.. number_of_nodes_to_process {
        //     for k in 0.. feat_count {
        //         let mut g_k = 0.0;
        //         for j in 0.. bin_count {
        //             let mut x2s = 0;
        //             for i in 0.. class_label_count {
        //                 x2s += x2[n][k][i][j];
        //             }
        //             g_k += x2s as f64 / y_partitioned[n][k][j] as f64;
        //         }
        //         g[n][k] = g_k;
        //     }
        //     ans[n] = argmax(&g[n].clone())?;
        // }
        // return Ok(ans);


        // ///// END TEST ////////////////////////////






        // At this point we have all of our x, x^2 and y values. Now we can start calculation gini numerators/denominators
        let mut sum_of_x2_j =  vec![vec![vec![0; bin_count]; feat_count]; number_of_nodes_to_process];

        // let mut d_exclude_j = vec![vec![vec![Wrapping(0); bin_count]; feat_count]; number_of_nodes_to_process];
        // let mut d_include_j = vec![vec![]; number_of_nodes_to_process];

        let mut d_exclude_j = vec![];
        let mut d_include_j = vec![];

        // create vector of the 0 and 1 values for j to set us up for batch multiplicaiton.
        // also, sum all of the x^2 values over i, and push these sums over i to a vector
        // to batch multiply with the y_without_j values.
        for n in 0..number_of_nodes_to_process {

            let mut y_vals_include_j = vec![vec![]; feat_count];

            for k in 0..feat_count {

                let mut y_vals_exclude_j = vec![vec![]; bin_count];

                for j in 0.. bin_count {
                    
                    // at the j'th index of the vector, we need to 
                    // push all values at indeces that are not equal to j onto it
                    // not 100% sure about this..
                    for not_j in 0.. bin_count {
                        if j != not_j {
                            y_vals_exclude_j[j].push(y_partitioned[n][k][not_j]);
                        }
                    }

                    y_vals_include_j[k].push(y_partitioned[n][k][j]);
                    

                    let mut sum_j_values = 0;
        
                    for i in 0..class_label_count {
                        sum_j_values += x2[n][k][i][j];
                    }
                    
                    sum_of_x2_j[n][k][j] = sum_j_values;
                }
                d_exclude_j.extend(y_vals_exclude_j);
                // can be far better optimized. Named 'D' after De'Hooghs variable
                // d_exclude_j[n][k] = protocol::pairwise_mult_zq(&y_vals_exclude_j, ctx).unwrap();
                // println!("exclude {:?}", protocol::open(&d_exclude_j[n][k], ctx).unwrap()); //test
                // 
                // d_exclude_j.append(y_vals_exclude_j); what we should do?
            }
            d_include_j.extend(y_vals_include_j);
            //println!("{:?}", d_include_j);
            //d_include_j[n] = protocol::pairwise_mult_zq(&y_vals_include_j, ctx).unwrap();
            // println!("include {:?}", protocol::open(&d_include_j[n], ctx).unwrap()); //test
            // d_include_j.append(y_vals_include_j); what we should do?
        }

        let mut sum_of_x2_j_flattend = vec![];

        for n in 0.. number_of_nodes_to_process {
            for k in 0.. feat_count {
                sum_of_x2_j_flattend.append(&mut sum_of_x2_j[n][k].clone());
            }
        } 




        // ////////////////// TEST //////////////////// failed...?
        // // returns correct result IF  we don't multiply y values b alpha and then add 1. However,
        // // still returns wrong answer if we don't do the former operations, but let the algorithm run to its end
        // // which is strange.
        // let mut ans = vec![0; number_of_nodes_to_process];
        // let mut g = vec![vec![0.0; feat_count]; number_of_nodes_to_process];

        // for n in 0.. number_of_nodes_to_process {
        //     for k in 0.. feat_count {
        //         let mut g_k = 0.0;
        //         for j in 0.. bin_count {
        //             let mut x2s = sum_of_x2_j[n][k][j];
        //             g_k += x2s as f64 / y_partitioned[n][k][j] as f64;
        //         }
        //         g[n][k] = g_k;
        //     }
        //     ans[n] = argmax(&g[n].clone())?;
        // }
        // return Ok(ans);

        // ///// END TEST ////////////////////////////





        let d_exclude_j = protocol_par(&d_exclude_j);
        let d_include_j = protocol_par(&d_include_j);

        let gini_numerators_values_flat_unsummed = protocol_mult(&d_exclude_j, &sum_of_x2_j_flattend);

        // println!("{}", d_exclude_j_flattend.len()); //test
        // println!("{}", sum_of_x2_j_flattend.len()); //test

        for v in 0.. gini_numerators_values_flat_unsummed.len() / bin_count {
            for j in 0.. bin_count {
                gini_numerators[v] += gini_numerators_values_flat_unsummed[v * bin_count + j];
            }
        }

        // create denominators
        let mut gini_denominators: Vec<usize> = d_include_j;

        // println!("{}: {:?}", gini_numerators.len(), gini_numerators); //test
        // println!("{}: {:?}", gini_denominators.len(), gini_denominators); //test



        // //////////// TEST //////////////// RETURNS INCORRECT RESULT (regardless of alpha value)

        // let mut gini_ratio = vec![];

        // for v in 0.. gini_numerators.len() / feat_count {
        //     let mut tmp = vec![];
        //     for n in 0..feat_count {
        //         let idx = v * feat_count + n;
        //         tmp.push(gini_numerators[idx] as f64 / gini_denominators[idx] as f64)
        //     }
        //     gini_ratio.push(argmax(&tmp)?);
        // }
        // return Ok(gini_ratio);


        // //// END TEST //////////// 










        // for n in 0.. number_of_nodes_to_process {
        //     for k in 0.. feat_count {
        //         let mut g_k = 0.0;
        //         for j in 0.. bin_count {
        //             let mut x2s = sum_of_x2_j[n][k][j];
        //             g_k += x2s as f64 / y_partitioned[n][k][j] as f64;
        //         }
        //         g[n][k] = g_k;
        //     }
        //     ans[n] = argmax(&g[n].clone())?;
        // }

        ////////////////// TEST //////////////////// 
        let mut j_mult = vec![vec![1 as usize; feat_count]; number_of_nodes_to_process];
        
        let mut gini_numerators_j = vec![0 as usize; feat_count * number_of_nodes_to_process * bin_count];
        let mut gini_denominators_j = vec![0 as usize; feat_count * number_of_nodes_to_process * bin_count];
        let mut common_denominator: Vec<Vec<usize>> = vec![];
        let mut common_denominator_withoutj: Vec<Vec<usize>> = vec![];
    
        for n in 0.. number_of_nodes_to_process {
            for k in 0.. feat_count {
                let mut common_dem_values = vec![];
                for j in 0.. bin_count {
                    gini_numerators_j[n * feat_count * bin_count + k * bin_count + j] = sum_of_x2_j[n][k][j];

                    common_dem_values.push(y_partitioned[n][k][j].clone());

                    let mut tmp = vec![];
                    for not_j in 0.. bin_count {
                        if j != not_j {
                            tmp.push(y_partitioned[n][k][not_j])
                        }
                    }     

                    common_denominator_withoutj.push(tmp);      
                }

                common_denominator.push(common_dem_values);

            }
        }

        let common_denominator_withoutj = protocol_par(&common_denominator_withoutj);
        let common_denominator = protocol_par(&common_denominator);
        let gini_numerators_unsummed = protocol_mult(&gini_numerators_j, &common_denominator_withoutj);

        for v in 0.. feat_count * number_of_nodes_to_process {
            gini_numerators[v] = gini_numerators_unsummed[bin_count * v.. bin_count * (v + 1)].to_vec().iter().sum();
        }
        
        let gini_denominators = common_denominator;
        ///// END TEST ////////////////////////////


        /////////////////////////////////////////// COMPUTE ARGMAX ///////////////////////////////////////////

        let mut current_length = feat_count;

        let mut logical_partition_lengths = vec![];

        let mut new_numerators = gini_numerators.clone();
        let mut new_denominators = gini_denominators.clone();

        // this will allow us to calculate the arg_max
        let mut past_assignments = vec![];

        loop {

            let odd_length = current_length % 2 == 1;
        
            let mut forgotten_numerators = vec![];
            let mut forgotten_denominators = vec![];
    
            let mut current_numerators = vec![];
            let mut current_denominators = vec![];
    
            // if its of odd length, make current nums/dems even lengthed, and store the 'forgotten' values
            // it should be guarenteed that we have numerators/denonminators of even length
            if odd_length {
                for n in 0 .. number_of_nodes_to_process {
                    current_numerators.append(&mut new_numerators[n * (current_length).. (n + 1) * (current_length - 1) + n].to_vec());
                    current_denominators.append(&mut new_denominators[n * (current_length).. (n + 1) * (current_length - 1) + n].to_vec());
    
                    forgotten_numerators.push(new_numerators[(n + 1) * (current_length - 1) + n]);
                    forgotten_denominators.push(new_denominators[(n + 1) * (current_length - 1) + n]);
    
                }
            } else {
                current_numerators = new_numerators.clone();
                current_denominators = new_denominators.clone();
            }
    
            // if denominators were originally indexed as d_1, d_2, d_3, d_4... then they are now represented
            // as d_2, d_1, d_4, d_3... This is helpful for comparisons
            let mut current_denominators_flipped = vec![];
            for v in 0 .. current_denominators.len()/2 {
                current_denominators_flipped.push(current_denominators[2 * v + 1]);
                current_denominators_flipped.push(current_denominators[2 * v]);
            }
    
            let product = protocol_mult(&current_numerators, &current_denominators_flipped);
    
            let mut l_operands = vec![];
            let mut r_operands = vec![];
    
            // left operands should be of the form n_1d_2, n_3d_4...
            // right operands should be of the form n_2d_1, n_4d_3...
            for v in 0..product.len()/2 {
                l_operands.push(product[2 * v]);
                r_operands.push(product[2 * v + 1]);
            }
    
            // read this as "left is greater than or equal to right." value in array will be [1] if true, [0] if false.
            let l_geq_r = protocol_geq(&l_operands, &r_operands);  
    
            // read this is "left is less than right"
            let l_lt_r: Vec<usize> = l_geq_r.iter().map(|x| 1 - x).collect();
    
            // grab the original values 
            let mut values = current_numerators.clone();
            values.append(&mut current_denominators.clone());
    
            // For the next iteration
            let mut assignments = vec![];
            // For record keeping
            let mut gini_assignments = vec![];
    
            // alternate the left/right values to help to cancel out the original values
            // that lost in their comparison
            for v in 0..l_geq_r.len() {
                assignments.push(l_geq_r[v]);
                assignments.push(l_lt_r[v]);
    
                gini_assignments.push(l_geq_r[v]);
                gini_assignments.push(l_lt_r[v]);
    
                let size = current_length/2;
                if odd_length && ((v + 1) % size == 0) {
                    gini_assignments.push(1);
                }
            }
    
            logical_partition_lengths.push(current_length);
            past_assignments.push(gini_assignments); 
    
            // EXIT CONDITION
            if 1 == (current_length/2) + (current_length % 2) {break;}
    
            assignments.append(&mut assignments.clone());
    
            let comparison_results = protocol_mult(&values, &assignments);
    
            new_numerators = vec![];
            new_denominators = vec![];
    
            // re-construct new_nums and new_dems
            for v in 0.. values.len()/4 {

                new_numerators.push(comparison_results[2 * v] + comparison_results[2 * v + 1]);
                new_denominators.push(comparison_results[values.len()/2 + 2 * v] + comparison_results[values.len()/2 + 2 * v + 1]);

                if odd_length && ((v + 1) % (current_length/2) == 0) {
                    // ensures no division by 0
                    let divisor = if current_length > 1 {current_length/2} else {1};
                    new_numerators.push(forgotten_numerators[(v/divisor)]);
                    new_denominators.push(forgotten_denominators[(v/divisor)]);
                }
    
            }
    
            current_length = (current_length/2) + (current_length % 2);
        }

        // calculates flat arg_max in a tournament bracket style
        for v in (1..past_assignments.len()).rev() {

            if past_assignments[v].len() == past_assignments[v - 1].len() {
                past_assignments[v - 1] = protocol_mult(&past_assignments[v - 1], &past_assignments[v]);
                continue;
            }

            let mut extended_past_assignment_v = vec![];
            for w in 0.. past_assignments[v].len() {
                if ((w + 1) % logical_partition_lengths[v] == 0) && ((logical_partition_lengths[v - 1] % 2) == 1) {
                    extended_past_assignment_v.push(past_assignments[v][w]);
                    continue;
                }
                extended_past_assignment_v.push(past_assignments[v][w]);
                extended_past_assignment_v.push(past_assignments[v][w]);
            }
            past_assignments[v - 1] = protocol_mult(&past_assignments[v - 1], &extended_past_assignment_v);
        }

        // un-flatten arg_max
        for n in 0.. number_of_nodes_to_process {
            if feat_count == 1 { // if there is only one attr count
                gini_arg_max.push(vec![1 as usize]);
            } else {
                gini_arg_max.push(past_assignments[0][n * feat_count.. (n + 1) * feat_count].to_vec());
            }
        }

        // ADDED: un-OHE argmax
        let mut tmp = vec![];
        for k in 0.. feat_count {
            tmp.push(k);
        }

        let mut new_gini = vec![];

        for argm in gini_arg_max {
            new_gini.push(protocol_dot(&tmp.clone(), &argm))
        }
        assert_eq!(new_gini.len(), number_of_nodes_to_process);

        Ok(new_gini)
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

                let mut valid_vote = false;

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

                    // if valid
                    if tree[current_node].frequencies[0] != 0 || tree[current_node].frequencies[1] != 0 || d == ctx.max_depth -1 {
                        // 'argmax'
                        valid_vote = true;
                        if tree[current_node].frequencies[0] < tree[current_node].frequencies[1] {
                            votes[1] += 1;
                        } else {
                            votes[0] += 1;
                        }
                        break;
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