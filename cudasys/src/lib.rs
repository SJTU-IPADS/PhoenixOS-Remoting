#![allow(non_snake_case, non_upper_case_globals, non_camel_case_types)]
include!("bindings/cuda_runtime.rs");

#[cfg(test)]
mod tests {
    use super::*;

    // This should work without GPU
    #[test]
    fn get_version() {
        let mut version: i32 = 0;
        let result = unsafe { cudaDriverGetVersion(&mut version as *mut i32) };
        if result != cudaError::cudaSuccess {
            panic!("Cannot get driver version: ERROR={:?}", result);
        }
        println!("Version = {}", version);
    }
}
