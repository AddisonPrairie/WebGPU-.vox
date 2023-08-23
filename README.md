# WebGPU-.vox

This is a work in progress voxel model path tracer built with webGPU. It loads MagicaVoxel .vox models and path traces them.<br>
![](https://live.staticflickr.com/65535/53136225473_a2d66e2962_b.jpg)<br>
An example model from [this repository](https://github.com/ephtracy/voxel-model/tree/master). The model uses a lambert diffuse BRDF while the floor uses a Cook-Torrance BRDF with GGX-Smith microfacet distribution/shadowing and VNDF importance sampling. Was rendered with 512 spp in about 5 seconds on a laptop RTX 2060.
