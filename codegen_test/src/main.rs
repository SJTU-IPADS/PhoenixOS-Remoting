extern crate codegen;
extern crate network;

use codegen::{gen_deserialize, gen_serialize};
use network::*;

gen_serialize!("basic_s", "usize", "f64", "bool");
gen_deserialize!("basic_des", "usize", "f64", "bool");

fn main() { 
    let mut buffer = RawBuffer::new(64);
    println!("Before serialize: {:?}", buffer);
    basic_s(&mut buffer, &12,&1.2, &true).expect("succeed");    
    println!("After serialize: {:?}", buffer);

    let mut val: usize = 0;
    let mut val2: f64 = 0.0;
    let mut val3: bool = false;
    basic_des(&mut buffer, &mut val, &mut val2, &mut val3).expect("succeed");

    println!("After deserialize: {:?}, {:?}, {:?}", val, val2, val3);

    assert_eq!(val, 12);
    assert_eq!(val2, 1.2);
    assert_eq!(val3, true);
}