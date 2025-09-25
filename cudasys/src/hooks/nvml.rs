use crate::types::nvml::*;
use codegen::cuda_hook;
use std::os::raw::*;

#[cuda_hook(proc_id = 991000, async_api = false)]
fn nvmlInit_v2() -> nvmlReturn_t;

#[cuda_hook(proc_id = 991001)]
fn nvmlDeviceGetCount_v2(deviceCount: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991002, async_api = false)]
fn nvmlInitWithFlags(flags: c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991003, async_api = false)]
fn nvmlShutdown() -> nvmlReturn_t;

#[cuda_hook(proc_id = 991004)]
fn nvmlDeviceGetHandleByIndex_v2(index: c_uint, device: *mut nvmlDevice_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991005)]
fn nvmlDeviceGetName(
    device: nvmlDevice_t,
    #[host(output, len = length)] name: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991006)]
fn nvmlDeviceGetCudaComputeCapability(
    device: nvmlDevice_t,
    major: *mut c_int,
    minor: *mut c_int,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991007)]
fn nvmlDeviceGetP2PStatus(
    device1: nvmlDevice_t,
    device2: nvmlDevice_t,
    p2pIndex: nvmlGpuP2PCapsIndex_t,
    p2pStatus: *mut nvmlGpuP2PStatus_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991008, async_api = false)]
fn nvmlSystemGetDriverVersion(
    #[host(output, len = length)] version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991009, async_api = false)]
fn nvmlSystemGetNVMLVersion(
    version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991010, async_api = false)]
fn nvmlSystemGetCudaDriverVersion(
    cudaDriverVersion: *mut c_int,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991011, async_api = false)]
fn nvmlSystemGetCudaDriverVersion_v2(
    cudaDriverVersion: *mut c_int,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991012, async_api = false)]
fn nvmlInternalGetExportTable(
    ppExportTable: *mut *const c_void,
    #[device] pExportTableId: *mut c_void,
) -> nvmlReturn_t {
    'client_before_send: {
        // TODO: nvidia-smi seems to accept the return of an empty table
        return nvmlReturn_t::NVML_SUCCESS;
    }
}

#[cuda_hook(proc_id = 991013, async_api = false)]
fn nvmlEventSetCreate(set: *mut nvmlEventSet_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991014, async_api = false)]
fn nvmlDeviceGetIndex(
    device: nvmlDevice_t,
    index: *mut ::std::os::raw::c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991015, async_api = false)]
fn nvmlDeviceGetPciInfo_v3(
    device: nvmlDevice_t,
    pci: *mut nvmlPciInfo_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991016, async_api = false)]
fn nvmlDeviceGetPersistenceMode(
    device: nvmlDevice_t,
    mode: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991017, async_api = false)]
fn nvmlDeviceGetDisplayActive(
    device: nvmlDevice_t,
    isActive: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991018, async_api = false)]
fn nvmlDeviceGetEccMode(
    device: nvmlDevice_t,
    current: *mut nvmlEnableState_t,
    pending: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991019, async_api = false)]
fn nvmlDeviceGetFanSpeed(
    device: nvmlDevice_t,
    speed: *mut ::std::os::raw::c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991020, async_api = false)]
fn nvmlDeviceGetTemperature(
    device: nvmlDevice_t,
    sensorType: nvmlTemperatureSensors_t,
    temp: *mut ::std::os::raw::c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991021, async_api = false)]
fn nvmlDeviceGetPerformanceState(
    device: nvmlDevice_t,
    pState: *mut nvmlPstates_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991022, async_api = false)]
fn nvmlDeviceGetPowerUsage(
    device: nvmlDevice_t,
    power: *mut ::std::os::raw::c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991023, async_api = false)]
fn nvmlDeviceGetEnforcedPowerLimit(
    device: nvmlDevice_t,
    limit: *mut ::std::os::raw::c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991024, async_api = false)]
fn nvmlDeviceGetMemoryInfo_v2(
    device: nvmlDevice_t,
    memory: *mut nvmlMemory_v2_t,
) -> nvmlReturn_t {
    'server_before_execution: {
        unsafe{
            let mem_ptr = memory.as_mut_ptr();
            // TODO: Implements NVML_STRUCT_VERSION(Memory, 2), as detailed in nvml.h (https://github.com/NVIDIA/nvidia-settings/issues/78)
            (*mem_ptr).version = (std::mem::size_of::<nvmlMemory_v2_t>() | (2_usize << 24_usize)) as u32;
        }
    }
}

#[cuda_hook(proc_id = 991025, async_api = false)]
fn nvmlDeviceGetUtilizationRates(
    device: nvmlDevice_t,
    utilization: *mut nvmlUtilization_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991026, async_api = false)]
fn nvmlDeviceGetComputeMode(
    device: nvmlDevice_t,
    mode: *mut nvmlComputeMode_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991027, async_api = false)]
fn nvmlDeviceGetMigMode(
    device: nvmlDevice_t,
    currentMode: *mut ::std::os::raw::c_uint,
    pendingMode: *mut ::std::os::raw::c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991028, async_api = false)]
fn nvmlDeviceGetVirtualizationMode(
    device: nvmlDevice_t,
    pVirtualMode: *mut nvmlGpuVirtualizationMode_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991029, async_api = false)]
fn nvmlDeviceGetMaxMigDeviceCount(
    device: nvmlDevice_t,
    count: *mut ::std::os::raw::c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991030, async_api = false)]
fn nvmlDeviceIsMigDeviceHandle(
    device: nvmlDevice_t,
    isMigDevice: *mut ::std::os::raw::c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991031, async_api = false)]
fn nvmlDeviceValidateInforom(device: nvmlDevice_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991032, async_api = false)]
fn nvmlEventSetFree(set: nvmlEventSet_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991033, async_api = false)]
fn nvmlDeviceGetMemoryInfo(
    device: nvmlDevice_t,
    memory: *mut nvmlMemory_t,
) -> nvmlReturn_t;