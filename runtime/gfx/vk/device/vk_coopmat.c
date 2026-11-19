// VK_KHR_cooperative_matrix 16x8x16 fp16 subgroup MMA discovery and boot verification.
// Idea from NotRexed/Hookshot (GPL-3.0, docs/research.md): discover the device's per-lane coopmat element order with a
// row-major coopMatLoad probe and verify coopMatMulAdd against the PTX mma.m16n8k16 fragment layout before enabling
// .cm.spv compute variants.

#include <string.h>

#include "cfg.h"
#include "coopmat_order_spv.h"
#include "coopmat_test_spv.h"
#include "host.h"
#include "vk.h"

static uint16_t i32_to_f16(int v) {
    if (!v) return 0;
    uint16_t sign = v < 0 ? 0x8000u : 0u;
    uint32_t a = (uint32_t)(v < 0 ? -v : v);
    int msb = 31 - __builtin_clz(a);
    uint16_t exp = (uint16_t)((msb + 15) << 10);
    uint16_t mant = (uint16_t)(((a ^ (1u << msb)) << (10 - msb)) & 0x3ffu);
    return sign | exp | mant;
}

static uint32_t exp_a(uint32_t lane, uint32_t f) {
    uint32_t g = lane >> 2, t = lane & 3u;
    uint32_t row = g + ((f & 2u) ? 8u : 0u);
    uint32_t col = 2u * t + (f & 1u) + ((f & 4u) ? 8u : 0u);
    return row * 16u + col;
}

static uint32_t exp_b(uint32_t lane, uint32_t f) {
    uint32_t g = lane >> 2, t = lane & 3u;
    uint32_t row = 2u * t + (f & 1u) + ((f & 2u) ? 8u : 0u);
    return row * 8u + g;
}

static uint32_t exp_c(uint32_t lane, uint32_t f) {
    uint32_t g = lane >> 2, t = lane & 3u;
    uint32_t row = g + ((f & 2u) ? 8u : 0u);
    uint32_t col = 2u * t + (f & 1u);
    return row * 8u + col;
}

static int test_a_val(uint32_t idx) {
    uint32_t r = idx >> 4, k = idx & 15u;
    return (int)((r * 3u + k * 5u) % 7u) - 3;
}

static int test_b_val(uint32_t idx) {
    uint32_t k = idx >> 3, c = idx & 7u;
    return (int)((k * 2u + c * 3u) % 5u) - 2;
}

static int test_c_val(uint32_t idx) {
    uint32_t r = idx >> 3, c = idx & 7u;
    return (int)((r * 5u + c * 7u) % 11u) - 5;
}

static int test_d_val(uint32_t idx) {
    uint32_t r = idx >> 3, c = idx & 7u;
    int sum = test_c_val(idx);
    for (uint32_t k = 0; k < 16u; k++) sum += test_a_val(r * 16u + k) * test_b_val(k * 8u + c);
    return sum;
}

static uint32_t pack2_f16(int lo, int hi) {
    return (uint32_t)i32_to_f16(lo) | ((uint32_t)i32_to_f16(hi) << 16);
}

static int run_probe(VkCtx *c, VkPipelineLayout layout, VkDescriptorSet dset, VkCommandPool pool,
                     VkCommandBuffer cb, VkFence fence, const uint32_t *spv, size_t spv_bytes,
                     const VkSpecializationInfo *spec) {
    VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = spv_bytes, .pCode = spv};
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(c->device, &smi, NULL, &mod) != VK_SUCCESS) return 0;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rqc = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 32};
    VkComputePipelineCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .pNext = &rqc,
                  .flags = VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT |
                           VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = mod,
                  .pName = "main",
                  .pSpecializationInfo = spec},
        .layout = layout};
    VkPipeline pipe = VK_NULL_HANDLE;
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &pci, NULL, &pipe);
    vkDestroyShaderModule(c->device, mod, NULL);
    if (r != VK_SUCCESS || !pipe) return 0;

    vkResetCommandPool(c->device, pool, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dset, 0, NULL);
    vkCmdDispatch(cb, 1, 1, 1);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &dep);
    vkEndCommandBuffer(cb);

    VkSubmitInfo sub = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    vkResetFences(c->device, 1, &fence);
    vk_queue_lock();
    r = vkQueueSubmit(c->gfx_queue, 1, &sub, fence);
    vk_queue_unlock();
    int ok = r == VK_SUCCESS && vkWaitForFences(c->device, 1, &fence, VK_TRUE, 1000000000ull) == VK_SUCCESS;
    vkDestroyPipeline(c->device, pipe, NULL);
    return ok;
}

void vk_coopmat_init(VkCtx *c) {
    if (!c || !c->has_coopmat || vk_caps.sg_size != 32 ||
        !(vk_caps.sg_pin_stages & VK_SHADER_STAGE_COMPUTE_BIT) || !rt_knob("VK_COOPMAT", "1"))
        return;

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR fn =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
            c->instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    uint32_t n = 0;
    if (!fn || fn(c->phys, &n, NULL) != VK_SUCCESS || !n) return;
    if (n > 64) n = 64;
    VkCooperativeMatrixPropertiesKHR props[64];
    memset(props, 0, sizeof props);
    for (uint32_t i = 0; i < n; i++) props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    if (fn(c->phys, &n, props) != VK_SUCCESS) return;

    int found = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (props[i].MSize == 16 && props[i].NSize == 8 && props[i].KSize == 16 &&
            props[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR && props[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            props[i].CType == VK_COMPONENT_TYPE_FLOAT16_KHR && props[i].ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            props[i].scope == VK_SCOPE_SUBGROUP_KHR) {
            found = 1;
            break;
        }
    }
    if (!found) return;

    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = 4096,
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VmaAllocationCreateInfo aci = {.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                            VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                   .usage = VMA_MEMORY_USAGE_AUTO,
                                   .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    VmaAllocationInfo ai = {0};
    if (vmaCreateBuffer(c->vma, &bci, &aci, &buf, &alloc, &ai) != VK_SUCCESS || !ai.pMappedData) {
        if (buf) vmaDestroyBuffer(c->vma, buf, alloc);
        return;
    }

    VkDescriptorSetLayoutBinding bind = {.binding = 0,
                                         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         .descriptorCount = 1,
                                         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dsli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = 1,
                                            .pBindings = &bind};
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(c->device, &dsli, NULL, &dsl) != VK_SUCCESS) goto out;

    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                      .setLayoutCount = 1,
                                      .pSetLayouts = &dsl};
    if (vkCreatePipelineLayout(c->device, &pli, NULL, &layout) != VK_SUCCESS) goto out;

    VkDescriptorPoolSize psize = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .maxSets = 1,
                                      .poolSizeCount = 1,
                                      .pPoolSizes = &psize};
    if (vkCreateDescriptorPool(c->device, &dpi, NULL, &dpool) != VK_SUCCESS) goto out;

    VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = dpool,
                                        .descriptorSetCount = 1,
                                        .pSetLayouts = &dsl};
    VkDescriptorSet dset = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(c->device, &dsai, &dset) != VK_SUCCESS) goto out;

    VkDescriptorBufferInfo dbi = {.buffer = buf, .offset = 0, .range = 4096};
    VkWriteDescriptorSet wds = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = dset,
                                .dstBinding = 0,
                                .descriptorCount = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .pBufferInfo = &dbi};
    vkUpdateDescriptorSets(c->device, 1, &wds, 0, NULL);

    VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                   .queueFamilyIndex = c->gfx_family};
    if (vkCreateCommandPool(c->device, &cpi, NULL, &cpool) != VK_SUCCESS) goto out;

    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                        .commandPool = cpool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                        .commandBufferCount = 1};
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(c->device, &cbai, &cb) != VK_SUCCESS) goto out;

    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(c->device, &fci, NULL, &fence) != VK_SUCCESS) goto out;

    memset(ai.pMappedData, 0, 4096);
    uint16_t *in_a = (uint16_t *)ai.pMappedData;
    uint16_t *in_bc = (uint16_t *)ai.pMappedData + 256;
    for (int i = 0; i < 256; i++) in_a[i] = i32_to_f16(i);
    for (int i = 0; i < 128; i++) in_bc[i] = i32_to_f16(i);

    if (!run_probe(c, layout, dset, cpool, cb, fence, coopmat_order_spv, sizeof coopmat_order_spv, NULL)) goto out;

    const uint32_t *ord = (const uint32_t *)ai.pMappedData + 256;
    uint32_t order_a[8] = {0}, order_b[4] = {0}, order_c[4] = {0};
    uint32_t seen_a = 0, seen_b = 0, seen_c = 0;

    for (uint32_t f = 0; f < 8u; f++) {
        for (uint32_t idx = 0; idx < 8u; idx++) {
            int match = 1;
            for (uint32_t lane = 0; lane < 32u; lane++)
                if (ord[lane * 16u + idx] != exp_a(lane, f)) { match = 0; break; }
            if (match) { order_a[f] = idx; seen_a |= 1u << idx; break; }
        }
    }
    for (uint32_t f = 0; f < 4u; f++) {
        for (uint32_t idx = 0; idx < 4u; idx++) {
            int match = 1;
            for (uint32_t lane = 0; lane < 32u; lane++)
                if (ord[lane * 16u + 8u + idx] != exp_b(lane, f)) { match = 0; break; }
            if (match) { order_b[f] = idx; seen_b |= 1u << idx; break; }
        }
        for (uint32_t idx = 0; idx < 4u; idx++) {
            int match = 1;
            for (uint32_t lane = 0; lane < 32u; lane++)
                if (ord[lane * 16u + 12u + idx] != exp_c(lane, f)) { match = 0; break; }
            if (match) { order_c[f] = idx; seen_c |= 1u << idx; break; }
        }
    }
    if (seen_a != 0xffu || seen_b != 0xfu || seen_c != 0xfu) goto out;

    uint32_t spec_vals[16];
    VkSpecializationMapEntry spec_entries[16];
    for (uint32_t i = 0; i < 8u; i++) spec_vals[i] = order_a[i];
    for (uint32_t i = 0; i < 4u; i++) spec_vals[8u + i] = order_b[i];
    for (uint32_t i = 0; i < 4u; i++) spec_vals[12u + i] = order_c[i];
    for (uint32_t i = 0; i < 16u; i++)
        spec_entries[i] = (VkSpecializationMapEntry){.constantID = 1024u + i, .offset = i * 4u, .size = 4u};
    VkSpecializationInfo spec = {.mapEntryCount = 16,
                                 .pMapEntries = spec_entries,
                                 .dataSize = sizeof spec_vals,
                                 .pData = spec_vals};

    memset(ai.pMappedData, 0, 4096);
    uint32_t *words = (uint32_t *)ai.pMappedData;
    for (uint32_t lane = 0; lane < 32u; lane++) {
        uint32_t *in_l = words + lane * 8u;
        for (uint32_t k = 0; k < 4u; k++)
            in_l[k] = pack2_f16(test_a_val(exp_a(lane, 2u * k)), test_a_val(exp_a(lane, 2u * k + 1u)));
        for (uint32_t k = 0; k < 2u; k++)
            in_l[4u + k] = pack2_f16(test_b_val(exp_b(lane, 2u * k)), test_b_val(exp_b(lane, 2u * k + 1u)));
        for (uint32_t k = 0; k < 2u; k++)
            in_l[6u + k] = pack2_f16(test_c_val(exp_c(lane, 2u * k)), test_c_val(exp_c(lane, 2u * k + 1u)));
    }

    if (!run_probe(c, layout, dset, cpool, cb, fence, coopmat_test_spv, sizeof coopmat_test_spv, &spec)) goto out;

    for (uint32_t lane = 0; lane < 32u; lane++) {
        uint32_t d0 = pack2_f16(test_d_val(exp_c(lane, 0)), test_d_val(exp_c(lane, 1)));
        uint32_t d1 = pack2_f16(test_d_val(exp_c(lane, 2)), test_d_val(exp_c(lane, 3)));
        if (words[256u + lane * 2u] != d0 || words[256u + lane * 2u + 1u] != d1) goto out;
    }

    memcpy(vk_caps.mma_order_a, order_a, sizeof order_a);
    memcpy(vk_caps.mma_order_b, order_b, sizeof order_b);
    memcpy(vk_caps.mma_order_c, order_c, sizeof order_c);
    vk_caps.coop_mma = 1;
    rt_log("vk caps: cooperative matrix 16x8x16 fp16: enabled (order a=%u%u%u%u%u%u%u%u b=%u%u%u%u c=%u%u%u%u)",
           order_a[0], order_a[1], order_a[2], order_a[3], order_a[4], order_a[5], order_a[6], order_a[7],
           order_b[0], order_b[1], order_b[2], order_b[3], order_c[0], order_c[1], order_c[2], order_c[3]);

out:
    if (fence) vkDestroyFence(c->device, fence, NULL);
    if (cpool) vkDestroyCommandPool(c->device, cpool, NULL);
    if (dpool) vkDestroyDescriptorPool(c->device, dpool, NULL);
    if (layout) vkDestroyPipelineLayout(c->device, layout, NULL);
    if (dsl) vkDestroyDescriptorSetLayout(c->device, dsl, NULL);
    vmaDestroyBuffer(c->vma, buf, alloc);
}
