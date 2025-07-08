use cudasys::types::cudnn::*;
use std::os::raw::*;

#[no_mangle]
pub extern "C" fn cudnnGetErrorString(
    error_status: cudnnStatus_t,
) -> *const c_char {
    log::debug!(target: "cudnnGetErrorString", "{error_status:?}");
    let result = format!("{error_status:?} ({})", error_status as u32);
    let c_str = std::ffi::CString::new(result).unwrap();
    c_str.into_raw()
}

#[no_mangle]
extern "C" fn cudnnBackendGetAttribute(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t {
    let mut element_count = 0;
    let elementCount = match elementCount.is_null() {
        true => &raw mut element_count,
        false => elementCount,
    };
    if arrayOfElements.is_null() {
        super::cudnn_hijack::cudnnBackendGetAttributeCount(
            descriptor,
            attributeName,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements,
        )
    } else if let cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR = attributeType {
        super::cudnn_hijack::cudnnBackendGetAttributeDescriptors(
            descriptor,
            attributeName,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements.cast(),
        )
    } else {
        super::cudnn_hijack::cudnnBackendGetAttributeData(
            descriptor,
            attributeName,
            attributeType,
            requestedElementCount,
            elementCount,
            arrayOfElements.cast(),
        )
    }
}
