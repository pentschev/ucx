/**
 * Copyright (C) 2023-2026 NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucp/api/ucp.h>
#include <cuda.h>
#include <stdio.h>


#define CUDA_CALL(_func) \
    ({ \
        CUresult _cu_result = (_func); \
        const char *_error_string; \
        if (_cu_result != CUDA_SUCCESS) { \
            cuGetErrorString(_cu_result, &_error_string); \
            printf("%s failed: %s\n", #_func, _error_string); \
        } \
        _cu_result; \
    })


int main(int argc, char **argv)
{
    int ret = -EXIT_FAILURE;
    ucp_params_t ucp_params;
    ucp_mem_map_params_t mmap_params;
    ucp_mem_attr_t mem_attr;
    ucp_context_h ucp_context;
    ucp_mem_h memh = NULL;
    ucs_status_t status;
    CUcontext cu_context;
    CUdevice cuda_device;
    size_t total_device_mem;
    int cuda_device_ordinal;

    if (setenv("UCX_CUDA_COPY_RETAIN_PRIMARY_CTX", "y", 1) != 0) {
        printf("setenv failed\n");
        goto out;
    }

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_TAG;

    status = ucp_init(&ucp_params, NULL, &ucp_context);
    if (status != UCS_OK) {
        printf("ucp_init failed: %s\n", ucs_status_string(status));
        goto out;
    }

    if (CUDA_CALL(cuInit(0)) != CUDA_SUCCESS) {
        goto cleanup;
    }

    if (CUDA_CALL(cuCtxGetCurrent(&cu_context)) != CUDA_SUCCESS) {
        goto cleanup;
    }

    if (cu_context != NULL) {
        printf("failed: CUDA context is not NULL\n");
        goto cleanup;
    }

    mmap_params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                              UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                              UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_params.length      = 4096;
    mmap_params.flags       = UCP_MEM_MAP_ALLOCATE;
    mmap_params.memory_type = UCS_MEMORY_TYPE_CUDA;

    if ((CUDA_CALL(cuDeviceGet(&cuda_device, 0)) != CUDA_SUCCESS) ||
        (CUDA_CALL(cuDeviceTotalMem(&total_device_mem, cuda_device)) !=
         CUDA_SUCCESS)) {
        goto cleanup;
    }

    mmap_params.length = total_device_mem + 1;
    status             = ucp_mem_map(ucp_context, &mmap_params, &memh);
    if (status == UCS_OK) {
        printf("unexpectedly allocated more than total CUDA memory\n");
        goto unmap;
    }

    if ((CUDA_CALL(cuCtxGetCurrent(&cu_context)) != CUDA_SUCCESS) ||
        (cu_context != NULL)) {
        printf("failed: CUDA context leaked after allocation failure\n");
        goto cleanup;
    }

    mmap_params.length = 4096;
    status = ucp_mem_map(ucp_context, &mmap_params, &memh);
    if (status != UCS_OK) {
        printf("ucp_mem_map failed: %s\n", ucs_status_string(status));
        goto cleanup;
    }

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status = ucp_mem_query(memh, &mem_attr);
    if (status != UCS_OK) {
        printf("ucp_mem_query failed: %s\n", ucs_status_string(status));
        goto unmap;
    }

    if (CUDA_CALL(cuPointerGetAttribute(
                          &cuda_device_ordinal,
                          CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                          (CUdeviceptr)mem_attr.address)) !=
        CUDA_SUCCESS) {
        goto unmap;
    }

    if (cuda_device_ordinal < 0) {
        printf("failed: invalid CUDA device ordinal %d\n",
               cuda_device_ordinal);
        goto unmap;
    }

    printf("SUCCESS\n");
    ret = EXIT_SUCCESS;

unmap:
    status = ucp_mem_unmap(ucp_context, memh);
    if (status != UCS_OK) {
        printf("ucp_mem_unmap failed: %s\n", ucs_status_string(status));
        ret = -EXIT_FAILURE;
    }

    if ((CUDA_CALL(cuCtxGetCurrent(&cu_context)) != CUDA_SUCCESS) ||
        (cu_context != NULL)) {
        printf("failed: CUDA context leaked after unmap\n");
        ret = -EXIT_FAILURE;
    }
cleanup:
    ucp_cleanup(ucp_context);
out:
    return ret;
}
