extern crate csv;

use fixed::{types::extra::U10, FixedI64};
use std::{error::Error};
use std::fs::File;
use std::fs;
use rand::*;

pub fn matrix_csv_to_float_vec(filename: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {

    let file = File::open(filename)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)    
        .from_reader(file);

    let mut vec: Vec<Vec<f64>> = Vec::new(); 

    for entry in rdr.records() {

        let row: Vec<String> = entry?.deserialize(None)?;
        let row = row.iter().map(|e| e.parse::<f64>().unwrap()).collect::<Vec<f64>>();
        vec.push(row);

    }
    Ok(vec)

}

pub fn transpose<T: Clone>(v: &Vec<Vec<T>>) -> Result<Vec<Vec<T>>, Box<dyn Error>> {
    if v.is_empty() {return Ok(vec![])}
    Ok((0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect())
}

// pub fn transpose(mat: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {

//     let mut mat_t = vec![vec![0f64; mat.len() ] ; mat[0].len()];

//     for i in 0..mat.len() {
//         for j in 0..mat[0].len() {
//             mat_t[j][i] = mat[i][j] 
//         }
//     }

//     Ok(mat_t)

// }

pub fn truncate(x: &f64, decimal_precision: usize, use_lossy: bool) -> Result<f64, Box<dyn Error>> {
    if use_lossy {
        let val = (x * (2f64.powf(decimal_precision as f64) )).round() / (2f64.powf(decimal_precision as f64));
        Ok(val)
    } else {
        Ok(*x)
    }
}

pub fn get_ratios(num: usize, decimal_precision: usize, seed: usize, use_lossy: bool) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut rng = if seed > 0 {rand::StdRng::from_seed(&[seed])} else {rand::StdRng::new()?};
    let mut result = vec![];
    for _i in 0 .. num {
        result.push(truncate(&(rng.gen_range(0, 1 << decimal_precision) as f64/2f64.powf(decimal_precision as f64) as f64), decimal_precision, use_lossy)?);
    }
    Ok(result)
}

pub fn get_features(num: usize, attribute_count: usize, seed: usize) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut rng = if seed > 0 {rand::StdRng::from_seed(&[seed])} else {rand::StdRng::new()?};
    let mut result = vec![];
    for _i in 0 .. num {
        result.push(rng.gen_range(0, attribute_count));
    }
    Ok(result)
}