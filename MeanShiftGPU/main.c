// Copyright 2015 Atomic Object

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <OpenCL/opencl.h>

#include "mean_shift_point.cl.h"

//#define VERIFY_OPENCL_OUTPUT
//#define USE_CPU

static int verify_mean_shift(cl_float2 *points, cl_float2 *original_points,
                             size_t num_points, cl_float bandwidth,
                             cl_float2 *shifted_points);

int main (int argc, const char *argv[]) {
    dispatch_queue_t queue =

#ifdef USE_CPU
    NULL;
#else
    gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
#endif

    if (queue == NULL) {
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
        fprintf(stderr, "Warning: Running on CPU\n");
    }
    else {
        char name[128];

        cl_device_id gpu = gcl_get_device_id_with_dispatch_queue(queue);
        clGetDeviceInfo(gpu, CL_DEVICE_NAME, 128, name, NULL);
        fprintf(stderr, "Running on GPU %s\n", name);
    }

    const int NUM_VALUES = 128 * 160;
    const int BUFFER_SIZE = sizeof(cl_float2) * NUM_VALUES;
    const int MAX_ITERATIONS = 100;
    const cl_float BANDWIDTH = 3;

    // --- Host memory (normal addressable array allocations)

    cl_float2 *points = (cl_float2 *)malloc(BUFFER_SIZE);
    cl_float2 *original_points = (cl_float2 *)malloc(BUFFER_SIZE);
    cl_float2 *shifted_points = (cl_float2 *)malloc(BUFFER_SIZE);

    for (int i = 0; i < NUM_VALUES; i++) {
        points[i].x = (cl_float)i;
        points[i].y = (cl_float)i;
    }

    memcpy(original_points, points, BUFFER_SIZE);

    // --- OpenCL device buffers (these are not directly addressable)

    void *device_points = gcl_malloc(BUFFER_SIZE, points,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *device_original_points = gcl_malloc(BUFFER_SIZE, original_points,
                                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void *device_shifted_points = gcl_malloc(BUFFER_SIZE, NULL, CL_MEM_WRITE_ONLY);

    // --- Schedule work on the OpenCL dispatch queue

    __block int num_work_groups = 0;
    int iteration = 0;

    for (;;) {

        // Dispatch is normally async, but this demo doesn't have other
        // work to do while waiting for the GPU. Also be careful trying to
        // do other work if OpenCL is using the CPU--it doesn't leave any
        // unused cores by default.

        dispatch_sync(queue, ^{
            size_t work_group_size;
            gcl_get_kernel_block_workgroup_info(mean_shift_point_kernel,
                                                CL_KERNEL_WORK_GROUP_SIZE,
                                                sizeof(work_group_size),
                                                &work_group_size, NULL);
            cl_ndrange range = {
                1,
                { 0, 0, 0 },
                { NUM_VALUES, 0, 0 },
                { work_group_size, 0, 0 }
            };

            num_work_groups = NUM_VALUES / work_group_size;

            mean_shift_point_kernel(&range, (cl_float2 *)device_points,
                                    (cl_float2 *)device_original_points,
                                    NUM_VALUES, BANDWIDTH,
                                    (cl_float2 *)device_shifted_points);

            gcl_memcpy(shifted_points, device_shifted_points, BUFFER_SIZE);
        });

        if (++iteration < MAX_ITERATIONS) {
            memcpy(points, shifted_points, BUFFER_SIZE);

            gcl_free(device_points);
            device_points = gcl_malloc(BUFFER_SIZE, points,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        }
        else {
            break;
        }
    }

    fprintf(stderr, "%d Iterations on %d work groups: Mean shifted %d points\n",
            iteration, num_work_groups, NUM_VALUES);

    // --- Release the OpenCL structures

    gcl_free(device_points);
    gcl_free(device_original_points);
    gcl_free(device_shifted_points);

    dispatch_release(queue);

    int status = 0;
#ifdef VERIFY_OPENCL_OUTPUT
    if (!verify_mean_shift(points, original_points, NUM_VALUES, BANDWIDTH,
                           shifted_points))
    {
        fprintf(stderr, "Values were not computed properly!\n");
        status = 1;
    }
#endif

    for (int i = 0; i < NUM_VALUES; i++) {
        printf("%15.8f, %15.8f\n", shifted_points[i].x, shifted_points[i].y);
    }

    free(points);
    free(original_points);
    free(shifted_points);

    exit(status);
}

static cl_float euclidian_distance(cl_float2 p1, cl_float2 p2) {
    return sqrtf(powf(p1.x - p2.x, 2.f) +
                 powf(p1.y - p2.y, 2.f));
}

static cl_float gaussian_kernel(cl_float dist, cl_float bandwidth) {
    return (1.f / (bandwidth * sqrtf(2.f * (float)M_PI))) *
            expf(-0.5f * powf(dist / bandwidth, 2.f));
}

static int verify_mean_shift(cl_float2 *points, cl_float2 *original_points,
                             size_t num_points, cl_float bandwidth,
                             cl_float2 *shifted_points)
{
    for (int i = 0; i < num_points; i++) {
        cl_float2 shift = { 0, 0 };
        cl_float scale = 0;

        for (int j = 0; j < num_points; j++) {
            cl_float dist = euclidian_distance(points[i], original_points[j]);
            cl_float weight = gaussian_kernel(dist, bandwidth);

            shift.x += original_points[j].x * weight;
            shift.y += original_points[j].y * weight;
            scale += weight;
        }

        cl_float2 expected = { shift.x / scale, shift.y / scale };

        if (fabs(shifted_points[i].x - expected.x) > 0.01 ||
            fabs(shifted_points[i].y - expected.y) > 0.01)
        {
            fprintf(stdout, "Error: Element %d did not match expected output.\n", i);
            fprintf(stdout, "       Saw (%1.8f,%1.8f), expected (%1.8f,%1.8f)\n",
                    shifted_points[i].x, shifted_points[i].y, expected.x, expected.y);
            fflush(stdout);
            return 0;
        }
    }

    return 1;
}
