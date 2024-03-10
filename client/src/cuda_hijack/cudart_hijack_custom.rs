#![allow(non_snake_case)]
use super::*;

#[no_mangle]
pub extern "C" fn cudaMemcpy(
    dst: MemPtr,
    src: MemPtr,
    count: usize,
    kind: cudaMemcpyKind,
) -> cudaError_t {
    println!("[{}:{}] cudaMemcpy", std::file!(), std::line!());

    if cudaMemcpyKind::cudaMemcpyHostToHost == kind {
        unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, count);
        }
        return cudaError_t::cudaSuccess;
    }

    let proc_id = 7;
    let mut result: cudaError_t = Default::default();
    match proc_id.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to send proc_id: {:?}", e),
    }
    match dst.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to send dst: {:?}", e),
    }
    match src.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to send src: {:?}", e),
    }
    match count.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to send count: {:?}", e),
    }
    match kind.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to send kind: {:?}", e),
    }

    if cudaMemcpyKind::cudaMemcpyHostToDevice == kind {
        // transport [src; count] to device
        let data = unsafe { std::slice::from_raw_parts(src as *const u8, count) };
        match data.send(&mut (*CHANNEL_SENDER.lock().unwrap())) {
            Ok(()) => {}
            Err(e) => panic!("failed to send data: {:?}", e),
        }
    }

    match CHANNEL_SENDER.lock().unwrap().flush_out() {
        Ok(()) => {}
        Err(e) => panic!("failed to send: {:?}", e),
    }

    if cudaMemcpyKind::cudaMemcpyDeviceToHost == kind {
        // receive [dst; count] from device
        let data = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, count) };
        match data.recv(&mut (*CHANNEL_RECEIVER.lock().unwrap())) {
            Ok(()) => {}
            Err(e) => panic!("failed to receive data: {:?}", e),
        }
    }

    match result.recv(&mut (*CHANNEL_RECEIVER.lock().unwrap())) {
        Ok(()) => {}
        Err(e) => panic!("failed to receive result: {:?}", e),
    }
    return result;
}

// TODO: maybe we should understand the semantic diff of cudaMemcpyAsync&cudaMemcpy
#[no_mangle]
pub extern "C" fn cudaMemcpyAsync(
    dst: MemPtr,
    src: MemPtr,
    count: usize,
    kind: cudaMemcpyKind,
    _stream: cudaStream_t,
) -> cudaError_t {
    println!("[{}:{}] cudaMemcpyAsync", std::file!(), std::line!());
    cudaMemcpy(dst, src, count, kind)
}
