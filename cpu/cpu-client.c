#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <driver_types.h>

//For TCP socket
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "cpu-libwrap.h"
#include "cpu_rpc_prot.h"
#include "cpu-common.h"
#include "cpu-utils.h"

//static const char* LIBCUDA_PATH = "/lib64/libcuda.so";
const char* LIBCUDA_PATH = "/usr/local/cuda/lib64/libcudart.so";
void *so_handle = NULL;

CLIENT *clnt = NULL;

size_t kernelnum = 0;
kernel_info_t *infos = NULL;

INIT_SOCKTYPE

void __attribute__ ((constructor)) init_rpc(void)
{
    enum clnt_stat retval_1;
    int result_1;
    int_result result_2;
    char *printmessage_1_arg1 = "hello";
    int isock;
    struct sockaddr_un sock_un = {0};
    struct sockaddr_in sock_in = {0};

    switch (socktype) {
    case UNIX:
        printf("connecting via UNIX...\n");
        isock = RPC_ANYSOCK;
        sock_un.sun_family = AF_UNIX;
        strcpy(sock_un.sun_path, CD_SOCKET_PATH);
        clnt = clntunix_create(&sock_un, RPC_CD_PROG, RPC_CD_VERS, &isock, 0, 0);
        break;
    case TCP:
        printf("connecting via TCP...\n");
        isock = RPC_ANYSOCK;
        sock_in.sin_family = AF_INET;
        sock_in.sin_port = 0;
        inet_aton("127.0.0.1", &sock_in.sin_addr);

        clnt = clnttcp_create(&sock_in, RPC_CD_PROG, RPC_CD_VERS, &isock, 0, 0);
        break;
    case UDP:
        /* From RPCEGEN documentation: 
         * Warning: since UDP-based RPC messages can only hold up to 8 Kbytes
         * of encoded data, this transport cannot be used for procedures that 
         * take large arguments or return huge results. 
         * -> Sounds like UDP does not make sense for CUDA, because we need to
         *    be able to copy large memory chunks
         **/
        printf("UDP is not supported...\n");
        break;
    }

    if (clnt == NULL) {
        printf("error\n");
        exit (1);
    }

    retval_1 = printmessage_1(printmessage_1_arg1, &result_1, clnt);
    printf("return:%d\n", result_1);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (!cricketd_utils_parameter_size(&infos, &kernelnum)) {
        fprintf(stderr, "error while getting parameter sizes\n");
        exit(0);
    }
}

static void deinit_rpc(void)
{
    clnt_destroy (clnt);
}



void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid,
                            uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize)
{
    int result;
    enum clnt_stat retval_1;

    //printf("__cudaRegisterFunction(fatCubinHandle=%p, hostFun=%p, devFunc=%s, deviceName=%s, thread_limit=%d, tid=[%p], bid=[%p], bDim=[%p], gDim=[%p], wSize=%p)\n", fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);

    kernel_info_t *info = cricketd_utils_search_info(infos, kernelnum, (char*)deviceName);
    if (info == NULL) {
        fprintf(stderr, "error: request to register unknown function\n");
    }
    info->host_fun = (void*)hostFun;

    retval_1 = RPC_SUCCESS;//cuda_register_function_1((uint64_t)fatCubinHandle, (uint64_t)hostFun, deviceFun, (char*)deviceName, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
}

struct __fatCubin {
    uint32_t magic;
    uint32_t seq;
    uint64_t text;
    uint64_t data;
    uint64_t ptr;
    uint64_t ptr2;
    uint64_t zero;
};

void** __cudaRegisterFatBinary(void *fatCubin)
{
    ptr_result result;
    enum clnt_stat retval_1;

    struct __fatCubin *fat = (struct __fatCubin*)((fatCubin));
    struct rpc_fatCubin rpc_fat = {.magic = fat->magic,
                                   .seq   = fat->seq,
                                   .text  = fat->text,
                                   .data  = fat->data,
                                   .ptr   = fat->ptr,
                                   .ptr2  = fat->ptr2,
                                   .zero  = fat->zero};
    //printf("__cudaRegisterFatBinary(magic: %x, seq: %x, text: %lx, data: %lx, ptr: %lx, ptr2: %lx, zero: %lx\n",
    //       fat->magic, fat->seq, fat->text, fat->data, fat->ptr, fat->ptr2, fat->zero);
    retval_1 = RPC_SUCCESS;//cuda_register_fat_binary_1(rpc_fat, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
    if (result.err != 0) {
        return NULL;
    }
    return (void*)result.ptr_result_u.ptr;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
    int result;
    enum clnt_stat retval_1;

    //printf("__cudaRegisterFatBinaryEnd(fatCubinHandle=%p)\n", fatCubinHandle);

    retval_1 = RPC_SUCCESS;//cuda_register_fat_binary_end_1((uint64_t)fatCubinHandle, &result, clnt);
    if (retval_1 != RPC_SUCCESS) {
        clnt_perror (clnt, "call failed");
    }
}
