// Copyright 2015 Atomic Object

kernel void mean_shift_point(global float2 *points,
                             global float2 *original_points,
                             size_t num_points,
                             float bandwidth,
                             global float2 *shifted_points)
{
    float base_weight = 1. / (bandwidth * sqrt(2. * M_PI_F));
    float2 shift = { 0, 0 };
    float scale = 0;

    size_t i = get_global_id(0);

    for (size_t j = 0; j < num_points; j++) {
        float dist = distance(points[i], original_points[j]);
        float weight = base_weight * exp(-0.5f * pow(dist / bandwidth, 2.f));

        shift += original_points[j] * weight;
        scale += weight;
    }

    shifted_points[i] = shift / scale;
}
