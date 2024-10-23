use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use numpy::ndarray::{Array2, ArrayView2, Axis};
use numpy::ndarray::parallel::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};
use rayon::prelude::*;

#[pymodule]
fn hw7_EthanBrown<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    fn lbp(image: ArrayView2<'_, u8>, p: usize, r: f64) -> Array2<u8> {
        let (height, width) = (image.shape()[0], image.shape()[1]);
        let mut output = Array2::<u8>::zeros((height, width));
        let r_int = r.ceil() as usize;
        
        if r_int >= height / 2 || r_int >= width / 2 {
            panic!("Radius is too large for the image dimensions.");
        }
        let (del_ks, del_ls): (Vec<f64>, Vec<f64>) = (0..p).map(|k| {
            (r * (2.0 * std::f64::consts::PI * k as f64 / p as f64).cos(),
            r * (2.0 * std::f64::consts::PI * k as f64 / p as f64).sin())
        }).collect();
    
        // Process chunks of rows in parallel
        let chunk_size = 10;
        output.axis_chunks_iter_mut(Axis(0), chunk_size)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut chunk)| {
                // Iterate over each row in the chunk
                for (row_offset, mut row) in chunk.axis_iter_mut(Axis(0)).enumerate() {
                    let i = chunk_idx * chunk_size + row_offset + r_int;
                    if i >= height - r_int {
                        continue;
                    }
    
                    for j in r_int..width - r_int {
                        let center_value = image[[i, j]];
                        let mut binary_pattern: u64 = 0;

                        del_ks.iter().zip(del_ls.iter()).enumerate().for_each(|(idx, (del_k, del_l))| {
                            let k_float = i as f64 + del_k;
                            let l_float = j as f64 + del_l;
        
                            if k_float < 0.0 || l_float < 0.0 || k_float >= (height as f64) || l_float >= (width as f64) {
                                return;
                            }
        
                            let k_base = k_float.floor() as usize;
                            let l_base = l_float.floor() as usize;
        
                            let delta_k = k_float - k_base as f64;
                            let delta_l = l_float - l_base as f64;
        
                            let image_val_at_p = if delta_k < 0.001 && delta_l < 0.001 {
                                image[[k_base, l_base]] as f64
                            } else if delta_l < 0.001 {
                                (1.0 - delta_k) * image[[k_base, l_base]] as f64
                                    + delta_k * image[[k_base + 1, l_base]] as f64
                            } else if delta_k < 0.001 {
                                (1.0 - delta_l) * image[[k_base, l_base]] as f64
                                    + delta_l * image[[k_base, l_base + 1]] as f64
                            } else {
                                (1.0 - delta_k) * (1.0 - delta_l) * image[[k_base, l_base]] as f64
                                    + (1.0 - delta_k) * delta_l * image[[k_base, l_base + 1]] as f64
                                    + delta_k * delta_l * image[[k_base + 1, l_base + 1]] as f64
                                    + delta_k * (1.0 - delta_l) * image[[k_base + 1, l_base]] as f64
                            };
                            if image_val_at_p >= center_value as f64 {
                                binary_pattern |= 1 << idx;
                            }
                        });
                        
                        let num_switches = (0..p).fold(0, |acc, idx| {
                            acc + (((binary_pattern >> idx) & 1) != ((binary_pattern >> ((idx + 1) % p)) & 1)) as u8
                        });
                        if num_switches <= 2 {
                            row[j] = binary_pattern.count_ones() as u8;
                        } else {
                            row[j] = p as u8 + 1;
                        }
                    }
                }
            });
    
        output
    }

    // Define the LBP function that will be exposed to Python
    #[pyfn(m)]
    #[pyo3(name = "compute_lbp")]
    fn compute_lbp_py<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, u8>,
        p: usize,
        r: f64,
    ) -> Bound<'py, PyArray2<u8>> {
        let image = image.as_array();
        let result = lbp(image, p, r);
        result.into_pyarray_bound(py)
    }

    Ok(())
}