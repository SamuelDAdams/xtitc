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

pub fn get_features(num: usize, attribute_count: usize, seed: usize, mode: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    if mode.eq("with_replacement") {
        let mut rng = if seed > 0 {rand::StdRng::from_seed(&[seed])} else {rand::StdRng::new()?};
        let mut result = vec![];
        for _i in 0 .. num {
            result.push(rng.gen_range(0, attribute_count));
        }
        return Ok(result)
    }

    // else without_replacement
    let mut indices = vec![];
    
    for i in 0.. attribute_count {
        indices.push(i);
    }

    let mut rng = if seed > 0 {rand::StdRng::from_seed(&[seed])} else {rand::StdRng::new()?};
    
    rng.shuffle(&mut indices);


    Ok(indices[0..num].to_vec())

}

pub fn protocol_mult(a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
    let mut c = vec![];

    assert_eq!(a.len(), b.len());

    for i in 0.. a.len() {
        c.push(a[i] * b[i]);
    }

    c
}

pub fn protocol_geq(a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
    let mut c = vec![];

    assert_eq!(a.len(), b.len());

    for i in 0.. a.len() {
        if a[i] >= b[i] {
            c.push(1)
        }
        else {
            c.push(0)
        }
    }
    c
}

pub fn protocol_dot(a: &Vec<usize>, b: &Vec<usize>) -> usize {
    let mut c = 0;

    assert_eq!(a.len(), b.len());

    for i in 0.. a.len() {
        c += a[i] * b[i];

        // if a[i] > 1 || b[i] > 1 {
        //     println!("WARNING: DOT PRODUCT CONTAINS NON-ZERO/ONE VALUES. a[i] = {}, b[i] = {}", a[i], b[i]);
        // }
    }

    c
}

pub fn protocol_par(a: &Vec<Vec<usize>>) -> Vec<usize> {
    let mut c = vec![];
    for i in 0.. a.len() {
        if a[i].len() <= 1 {
            c.push(a[i][0]);
            continue
        }
        assert_eq!(a[i].len(), a[(i + 1) % a[i].len()].len());
        let mut prod = 1;
        for val in a[i].clone() {
            prod += val * prod;
        }
        c.push(prod);
    }
    c
}

