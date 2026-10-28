// rigsim Vulkan probe (stick): vendor-neutral reference for llama.cpp's Vulkan backend. One JSON object per line.
//   vkprobe list                                                  every device, its cooperative-matrix shapes
//   vkprobe stream  --device I --op read|write|copy [--mib 1024] [--reps 20]
//   vkprobe launch  --device I [--count 2000]                     dispatch cost: back-to-back, with barriers, submit
//   vkprobe fma     --device I --prec fp32|fp16 [--iters 4096]    vector FMA peak (no matrix units)
//   vkprobe gemm    --device I --type f16|f16f32|i8 --shape MxNxK [--reps 10]   cooperative matrices (matrix units)
//   vkprobe shared  --device I [--seconds 8] [--threads 0]        integrated GPU: RAM bandwidth CPU alone, GPU
//                                                                 alone, both at once
// The GEMM is a straightforward cooperative-matrix kernel (2x2 tiles per subgroup, loads from global memory): it
// shows what the matrix units reach through Vulkan without a tuned library, the same path llama.cpp's coopmat
// kernels take. "unsupported" results are results, not errors.
#define _GNU_SOURCE
#include <vulkan/vulkan.h>

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const uint32_t SPV_STREAM[] = {
#include "stream.inc"
};
static const uint32_t SPV_EMPTY[] = {
#include "empty.inc"
};
static const uint32_t SPV_FILL[] = {
#include "fill.inc"
};
static const uint32_t SPV_FMA32[] = {
#include "fma32.inc"
};
static const uint32_t SPV_FMA16[] = {
#include "fma16.inc"
};
#ifdef HAVE_CM_F16
static const uint32_t SPV_CM_F16[] = {
#include "cm_F16.inc"
};
#endif
#ifdef HAVE_CM_F16F32
static const uint32_t SPV_CM_F16F32[] = {
#include "cm_F16F32.inc"
};
#endif
#ifdef HAVE_CM_I8
static const uint32_t SPV_CM_I8[] = {
#include "cm_I8.inc"
};
#endif

#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    printf("{\"error\":\"%s\",\"vk_result\":%d,\"line\":%d}\n", #x, (int)r_, __LINE__); exit(2); } } while (0)

static const char* arg(int argc, char** argv, const char* key, const char* dflt) {
    for (int i = 2; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
    return dflt;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }
static double median(double* v, int n) { qsort(v, n, sizeof(double), cmpd); return v[n / 2]; }
static void jstr(const char* s) {                     // JSON string, escaped
    putchar('"');
    for (; *s; ++s) { if (*s == '"' || *s == '\\') putchar('\\'); if ((unsigned char)*s >= 0x20) putchar(*s); }
    putchar('"');
}

// ---- context --------------------------------------------------------------------------------------------------------
typedef struct {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qf;
    VkPhysicalDeviceProperties props; VkPhysicalDeviceMemoryProperties mem;
    VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkDescriptorPool dp; VkDescriptorSet ds;
    VkCommandPool cp; VkCommandBuffer cb; VkFence fence; VkQueryPool qp;
    int has_ts; double ts_ns; uint64_t ts_mask; uint32_t subgroup;
    int f16, i8, s16, s8, cm, mm;
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR cm_props;
} Ctx;

typedef struct { VkBuffer b; VkDeviceMemory m; VkDeviceSize size; } Buf;

static int has_ext(VkPhysicalDevice pd, const char* name) {
    uint32_t n = 0; vkEnumerateDeviceExtensionProperties(pd, NULL, &n, NULL);
    VkExtensionProperties* e = calloc(n, sizeof(*e)); vkEnumerateDeviceExtensionProperties(pd, NULL, &n, e);
    int found = 0; for (uint32_t i = 0; i < n; ++i) if (!strcmp(e[i].extensionName, name)) found = 1;
    free(e); return found;
}

static VkInstance make_instance(void) {
    uint32_t v = VK_API_VERSION_1_1;
    PFN_vkEnumerateInstanceVersion ev = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceVersion");
    if (ev) ev(&v);
    if (v > VK_API_VERSION_1_3) v = VK_API_VERSION_1_3;
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "rigsim-vkprobe", 1, NULL, 0, v};
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai, 0, NULL, 0, NULL};
    VkInstance inst; VK(vkCreateInstance(&ci, NULL, &inst)); return inst;
}

static const char* type_name(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}
static const char* comp_name(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16"; case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_SINT8_KHR: return "s8"; case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
        case VK_COMPONENT_TYPE_SINT32_KHR: return "s32"; case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
        default: return "other";
    }
}

static void open_device(Ctx* c, int index) {
    memset(c, 0, sizeof(*c));
    c->inst = make_instance();
    uint32_t n = 0; vkEnumeratePhysicalDevices(c->inst, &n, NULL);
    if (index < 0 || (uint32_t)index >= n) { printf("{\"error\":\"no device %d (%u found)\"}\n", index, n); exit(1); }
    VkPhysicalDevice* pds = calloc(n, sizeof(*pds)); vkEnumeratePhysicalDevices(c->inst, &n, pds);
    c->pd = pds[index]; free(pds);
    vkGetPhysicalDeviceProperties(c->pd, &c->props);
    vkGetPhysicalDeviceMemoryProperties(c->pd, &c->mem);
    // compute queue
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(c->pd, &nq, NULL);
    VkQueueFamilyProperties* qfp = calloc(nq, sizeof(*qfp)); vkGetPhysicalDeviceQueueFamilyProperties(c->pd, &nq, qfp);
    c->qf = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i) if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        if (c->qf == UINT32_MAX || (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { c->qf = i; c->ts_mask = qfp[i].timestampValidBits; }
        if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) break;      // the universal queue, as llama.cpp uses by default
    }
    free(qfp);
    if (c->qf == UINT32_MAX) { printf("{\"error\":\"no compute queue\"}\n"); exit(1); }
    c->has_ts = c->ts_mask > 0 && c->props.limits.timestampPeriod > 0;
    c->ts_ns = c->props.limits.timestampPeriod;
    c->ts_mask = c->ts_mask >= 64 ? ~0ULL : ((1ULL << c->ts_mask) - 1);
    // features
    VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg};
    vkGetPhysicalDeviceProperties2(c->pd, &p2);
    c->subgroup = sg.subgroupSize ? sg.subgroupSize : 32;
    int v12 = c->props.apiVersion >= VK_API_VERSION_1_2;
    int cm_ext = v12 && has_ext(c->pd, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, cm_ext ? &fcm : NULL};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, v12 ? (void*)&f11 : NULL};
    vkGetPhysicalDeviceFeatures2(c->pd, &f2);
    c->f16 = v12 && f12.shaderFloat16; c->i8 = v12 && f12.shaderInt8;
    c->s16 = v12 && f11.storageBuffer16BitAccess; c->s8 = v12 && f12.storageBuffer8BitAccess;
    c->mm = v12 && f12.vulkanMemoryModel; c->cm = cm_ext && fcm.cooperativeMatrix && c->mm;
    // enable only what we use
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR ecm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    ecm.cooperativeMatrix = c->cm;
    VkPhysicalDeviceVulkan12Features e12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, c->cm ? &ecm : NULL};
    e12.shaderFloat16 = c->f16; e12.shaderInt8 = c->i8; e12.storageBuffer8BitAccess = c->s8;
    e12.vulkanMemoryModel = c->mm; e12.vulkanMemoryModelDeviceScope = c->mm && f12.vulkanMemoryModelDeviceScope;
    VkPhysicalDeviceVulkan11Features e11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &e12};
    e11.storageBuffer16BitAccess = c->s16;
    VkPhysicalDeviceFeatures2 e2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, v12 ? (void*)&e11 : NULL};
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, c->qf, 1, &prio};
    const char* exts[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &e2, 0, 1, &qci, 0, NULL, c->cm ? 1u : 0u, exts, NULL};
    VK(vkCreateDevice(c->pd, &dci, NULL, &c->dev));
    vkGetDeviceQueue(c->dev, c->qf, 0, &c->q);
    if (c->cm) c->cm_props = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(c->inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    // three storage buffers + 16 bytes of push constants, shared by every pipeline
    VkDescriptorSetLayoutBinding b[3];
    for (int i = 0; i < 3; ++i) b[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, b};
    VK(vkCreateDescriptorSetLayout(c->dev, &dl, NULL, &c->dsl));
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &c->dsl, 1, &pr};
    VK(vkCreatePipelineLayout(c->dev, &pli, NULL, &c->pl));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
    VK(vkCreateDescriptorPool(c->dev, &dpi, NULL, &c->dp));
    VkDescriptorSetAllocateInfo dsa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, c->dp, 1, &c->dsl};
    VK(vkAllocateDescriptorSets(c->dev, &dsa, &c->ds));
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, c->qf};
    VK(vkCreateCommandPool(c->dev, &cpi, NULL, &c->cp));
    VkCommandBufferAllocateInfo cba = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, c->cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VK(vkAllocateCommandBuffers(c->dev, &cba, &c->cb));
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK(vkCreateFence(c->dev, &fi, NULL, &c->fence));
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2};
    if (c->has_ts) VK(vkCreateQueryPool(c->dev, &qpi, NULL, &c->qp));
}

static Buf make_buf(Ctx* c, VkDeviceSize size) {
    Buf b = {0}; b.size = size;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VK(vkCreateBuffer(c->dev, &bi, NULL, &b.b));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(c->dev, b.b, &r);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < c->mem.memoryTypeCount; ++i)                 // device-local first, anything allowed after
        if ((r.memoryTypeBits & (1u << i)) && (c->mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { type = i; break; }
    for (uint32_t i = 0; type == UINT32_MAX && i < c->mem.memoryTypeCount; ++i) if (r.memoryTypeBits & (1u << i)) type = i;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, r.size, type};
    VkResult res = vkAllocateMemory(c->dev, &ai, NULL, &b.m);
    if (res != VK_SUCCESS) { printf("{\"error\":\"allocate %.0f MiB\",\"vk_result\":%d}\n", size / 1048576.0, (int)res); exit(2); }
    VK(vkBindBufferMemory(c->dev, b.b, b.m, 0));
    return b;
}

static void bind(Ctx* c, Buf* x, Buf* y, Buf* o) {
    VkDescriptorBufferInfo bi[3] = {{x->b, 0, VK_WHOLE_SIZE}, {y->b, 0, VK_WHOLE_SIZE}, {o->b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, c->ds, 0, 0, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, bi, NULL};
    vkUpdateDescriptorSets(c->dev, 1, &w, 0, NULL);
}

static VkPipeline pipeline(Ctx* c, const uint32_t* spv, size_t bytes, const uint32_t* spec, uint32_t nspec) {
    VkShaderModuleCreateInfo si = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, bytes, spv};
    VkShaderModule m; VK(vkCreateShaderModule(c->dev, &si, NULL, &m));
    VkSpecializationMapEntry e[8];
    for (uint32_t i = 0; i < nspec; ++i) e[i] = (VkSpecializationMapEntry){i, i * 4, 4};
    VkSpecializationInfo sp = {nspec, e, nspec * 4, spec};
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nspec ? &sp : NULL},
        c->pl, VK_NULL_HANDLE, 0};
    VkPipeline p; VkResult r = vkCreateComputePipelines(c->dev, VK_NULL_HANDLE, 1, &ci, NULL, &p);
    vkDestroyShaderModule(c->dev, m, NULL);
    return r == VK_SUCCESS ? p : VK_NULL_HANDLE;
}

// Records `count` dispatches between two timestamps, submits, waits. Returns the GPU time (s), or the wall time if
// the queue has no timestamps. *wall gets the host-side submit-to-fence time.
static double run(Ctx* c, VkPipeline p, uint32_t gx, uint32_t gy, const uint32_t push[4], int count, int barrier, double* wall) {
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK(vkResetCommandBuffer(c->cb, 0));
    VK(vkBeginCommandBuffer(c->cb, &bi));
    if (c->has_ts) { vkCmdResetQueryPool(c->cb, c->qp, 0, 2); vkCmdWriteTimestamp(c->cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c->qp, 0); }
    vkCmdBindPipeline(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdBindDescriptorSets(c->cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pl, 0, 1, &c->ds, 0, NULL);
    vkCmdPushConstants(c->cb, c->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    for (int i = 0; i < count; ++i) {
        if (barrier && i) vkCmdPipelineBarrier(c->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        vkCmdDispatch(c->cb, gx, gy, 1);
    }
    if (c->has_ts) vkCmdWriteTimestamp(c->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c->qp, 1);
    VK(vkEndCommandBuffer(c->cb));
    VkSubmitInfo s = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &c->cb, 0, NULL};
    VK(vkResetFences(c->dev, 1, &c->fence));
    double t0 = now();
    VK(vkQueueSubmit(c->q, 1, &s, c->fence));
    VK(vkWaitForFences(c->dev, 1, &c->fence, VK_TRUE, UINT64_MAX));
    double w = now() - t0;
    if (wall) *wall = w;
    if (!c->has_ts) return w;
    uint64_t ts[2];
    VK(vkGetQueryPoolResults(c->dev, c->qp, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return (double)((ts[1] - ts[0]) & c->ts_mask) * c->ts_ns * 1e-9;
}

static void fill(Ctx* c, Buf* x, uint32_t mode, uint32_t seed) {
    static VkPipeline p = VK_NULL_HANDLE;
    if (!p) p = pipeline(c, SPV_FILL, sizeof(SPV_FILL), NULL, 0);
    Buf* none = x;
    bind(c, x, none, none);
    uint32_t words = (uint32_t)(x->size / 4), push[4] = {words, mode, seed, 0};
    run(c, p, 4096, 1, push, 1, 0, NULL);
}

static uint32_t groups_for(uint64_t n, uint32_t per) {
    uint64_t g = (n + per - 1) / per; return (uint32_t)(g > 65535 ? 65535 : (g ? g : 1));
}

static void head(Ctx* c, const char* mode) {
    printf("{\"mode\":\"%s\",\"device\":", mode); jstr(c->props.deviceName);
    printf(",\"device_type\":\"%s\",\"vendor_id\":%u,\"device_id\":%u,\"driver_version\":%u,\"api\":\"%u.%u.%u\"",
           type_name(c->props.deviceType), c->props.vendorID, c->props.deviceID, c->props.driverVersion,
           VK_API_VERSION_MAJOR(c->props.apiVersion), VK_API_VERSION_MINOR(c->props.apiVersion),
           VK_API_VERSION_PATCH(c->props.apiVersion));
    printf(",\"timer\":\"%s\"", c->has_ts ? "gpu timestamps" : "wall clock");
}

// ---- list -----------------------------------------------------------------------------------------------------------
static int list_mode(void) {
    VkInstance inst = make_instance();
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    vkDestroyInstance(inst, NULL);
    for (uint32_t i = 0; i < n; ++i) {
        Ctx c; open_device(&c, (int)i);
        head(&c, "list");
        uint64_t local = 0, total = 0;
        for (uint32_t h = 0; h < c.mem.memoryHeapCount; ++h) {
            total += c.mem.memoryHeaps[h].size;
            if (c.mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local += c.mem.memoryHeaps[h].size;
        }
        printf(",\"index\":%u,\"subgroup\":%u,\"f16\":%s,\"int8\":%s,\"device_local_bytes\":%llu,\"heap_bytes\":%llu,"
               "\"max_workgroup\":%u,\"coopmat\":[", i, c.subgroup, c.f16 ? "true" : "false", c.i8 ? "true" : "false",
               (unsigned long long)local, (unsigned long long)total, c.props.limits.maxComputeWorkGroupInvocations);
        if (c.cm && c.cm_props) {
            uint32_t k = 0; c.cm_props(c.pd, &k, NULL);
            VkCooperativeMatrixPropertiesKHR* p = calloc(k, sizeof(*p));
            for (uint32_t j = 0; j < k; ++j) p[j].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            c.cm_props(c.pd, &k, p);
            for (uint32_t j = 0; j < k; ++j)
                printf("%s{\"m\":%u,\"n\":%u,\"k\":%u,\"a\":\"%s\",\"b\":\"%s\",\"c\":\"%s\",\"r\":\"%s\",\"scope\":%d}", j ? "," : "",
                       p[j].MSize, p[j].NSize, p[j].KSize, comp_name(p[j].AType), comp_name(p[j].BType),
                       comp_name(p[j].CType), comp_name(p[j].ResultType), (int)p[j].scope);
            free(p);
        }
        printf("]}\n");
        vkDestroyDevice(c.dev, NULL); vkDestroyInstance(c.inst, NULL);
    }
    if (!n) printf("{\"mode\":\"list\",\"devices\":0}\n");
    return 0;
}

// ---- stream ---------------------------------------------------------------------------------------------------------
static int stream_mode(Ctx* c, int argc, char** argv) {
    const char* op = arg(argc, argv, "--op", "read");
    uint32_t opc = !strcmp(op, "read") ? 0 : !strcmp(op, "write") ? 1 : 2;
    uint64_t bytes = strtoull(arg(argc, argv, "--mib", "1024"), NULL, 10) << 20;
    if (bytes > c->props.limits.maxStorageBufferRange) bytes = c->props.limits.maxStorageBufferRange & ~(uint64_t)0xfffff;
    int reps = atoi(arg(argc, argv, "--reps", "20"));
    Buf x = make_buf(c, bytes), y = make_buf(c, bytes), o = make_buf(c, 4096);
    fill(c, &x, 0, 1);
    bind(c, &x, &y, &o);
    VkPipeline p = pipeline(c, SPV_STREAM, sizeof(SPV_STREAM), &opc, 1);
    uint32_t n = (uint32_t)(bytes / 16), push[4] = {n, 0, 0, 0}, g = groups_for(n, 256 * 4);
    for (int i = 0; i < 3; ++i) run(c, p, g, 1, push, 1, 0, NULL);
    double* t = calloc(reps, sizeof(double));
    for (int i = 0; i < reps; ++i) t[i] = run(c, p, g, 1, push, 1, 0, NULL);
    double moved = opc == 2 ? 2.0 * bytes : (double)bytes, med = median(t, reps);
    head(c, "stream");
    printf(",\"op\":\"%s\",\"bytes\":%.0f,\"reps\":%d,\"t_median_s\":%.9g,\"t_min_s\":%.9g,\"bytes_per_s\":%.6g}\n",
           op, moved, reps, med, t[0], moved / med);
    return 0;
}

// ---- dispatch cost --------------------------------------------------------------------------------------------------
static int launch_mode(Ctx* c, int argc, char** argv) {
    int count = atoi(arg(argc, argv, "--count", "2000"));
    Buf o = make_buf(c, 4096);
    bind(c, &o, &o, &o);
    VkPipeline p = pipeline(c, SPV_EMPTY, sizeof(SPV_EMPTY), NULL, 0);
    uint32_t push[4] = {0, 0, 0, 0};
    run(c, p, 1, 1, push, count, 1, NULL);
    double a[7], b[7], s[101], w;
    for (int i = 0; i < 7; ++i) a[i] = run(c, p, 1, 1, push, count, 0, NULL) / count;
    for (int i = 0; i < 7; ++i) b[i] = run(c, p, 1, 1, push, count, 1, NULL) / count;
    for (int i = 0; i < 101; ++i) { run(c, p, 1, 1, push, 1, 0, &w); s[i] = w; }
    head(c, "launch");
    printf(",\"count\":%d,\"per_dispatch_s\":%.6g,\"per_dispatch_barrier_s\":%.6g,\"submit_roundtrip_s\":%.6g}\n",
           count, median(a, 7), median(b, 7), median(s, 101));
    return 0;
}

// ---- vector FMA -----------------------------------------------------------------------------------------------------
static int fma_mode(Ctx* c, int argc, char** argv) {
    const char* prec = arg(argc, argv, "--prec", "fp32");
    int half = !strcmp(prec, "fp16");
    head(c, "fma");
    if (half && !c->f16) { printf(",\"prec\":\"fp16\",\"unsupported\":true}\n"); return 0; }
    uint32_t iters = (uint32_t)atoi(arg(argc, argv, "--iters", "4096")), groups = 4096;
    Buf o = make_buf(c, 4096);
    bind(c, &o, &o, &o);
    VkPipeline p = half ? pipeline(c, SPV_FMA16, sizeof(SPV_FMA16), NULL, 0) : pipeline(c, SPV_FMA32, sizeof(SPV_FMA32), NULL, 0);
    if (!p) { printf(",\"prec\":\"%s\",\"unsupported\":true,\"why\":\"pipeline\"}\n", prec); return 0; }
    uint32_t push[4] = {iters, 0, 0, 0};
    run(c, p, groups, 1, push, 1, 0, NULL);
    double t[5]; for (int i = 0; i < 5; ++i) t[i] = run(c, p, groups, 1, push, 1, 0, NULL);
    double ops = (double)groups * 256 * iters * 8 * 4 * 2, med = median(t, 5);
    printf(",\"prec\":\"%s\",\"ops\":%.0f,\"t_median_s\":%.9g,\"ops_per_s\":%.6g}\n", prec, ops, med, ops / med);
    return 0;
}

// ---- cooperative-matrix GEMM ----------------------------------------------------------------------------------------
static int gemm_mode(Ctx* c, int argc, char** argv) {
    const char* type = arg(argc, argv, "--type", "f16");
    unsigned m, n, k;
    if (sscanf(arg(argc, argv, "--shape", "4096x4096x4096"), "%ux%ux%u", &m, &n, &k) != 3) return 1;
    int reps = atoi(arg(argc, argv, "--reps", "10"));
    VkComponentTypeKHR ab, cc; const uint32_t* spv = NULL; size_t spv_bytes = 0; size_t es, cs; uint32_t fillmode;
    if (!strcmp(type, "f16")) { ab = VK_COMPONENT_TYPE_FLOAT16_KHR; cc = VK_COMPONENT_TYPE_FLOAT16_KHR; es = 2; cs = 2; fillmode = 1;
#ifdef HAVE_CM_F16
        spv = SPV_CM_F16; spv_bytes = sizeof(SPV_CM_F16);
#endif
    } else if (!strcmp(type, "f16f32")) { ab = VK_COMPONENT_TYPE_FLOAT16_KHR; cc = VK_COMPONENT_TYPE_FLOAT32_KHR; es = 2; cs = 4; fillmode = 1;
#ifdef HAVE_CM_F16F32
        spv = SPV_CM_F16F32; spv_bytes = sizeof(SPV_CM_F16F32);
#endif
    } else if (!strcmp(type, "i8")) { ab = VK_COMPONENT_TYPE_SINT8_KHR; cc = VK_COMPONENT_TYPE_SINT32_KHR; es = 1; cs = 4; fillmode = 2;
#ifdef HAVE_CM_I8
        spv = SPV_CM_I8; spv_bytes = sizeof(SPV_CM_I8);
#endif
    } else { fprintf(stderr, "type is f16, f16f32 or i8\n"); return 1; }
    head(c, "gemm");
    printf(",\"type\":\"%s\",\"m\":%u,\"n\":%u,\"k\":%u", type, m, n, k);
    if (!c->cm || !c->cm_props) { printf(",\"unsupported\":true,\"why\":\"no VK_KHR_cooperative_matrix\"}\n"); return 0; }
    if (!spv) { printf(",\"unsupported\":true,\"why\":\"shader not built (compiler)\"}\n"); return 0; }
    if ((es == 2 && !(c->f16 && c->s16)) || (es == 1 && !(c->i8 && c->s8))) { printf(",\"unsupported\":true,\"why\":\"no %s storage\"}\n", type); return 0; }
    uint32_t kk = 0; c->cm_props(c->pd, &kk, NULL);
    VkCooperativeMatrixPropertiesKHR* p = calloc(kk, sizeof(*p));
    for (uint32_t j = 0; j < kk; ++j) p[j].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    c->cm_props(c->pd, &kk, p);
    int pick = -1;
    for (uint32_t j = 0; j < kk; ++j)
        if (p[j].AType == ab && p[j].BType == ab && p[j].CType == cc && p[j].ResultType == cc && p[j].scope == VK_SCOPE_SUBGROUP_KHR)
            if (pick < 0 || (p[j].MSize == 16 && p[j].NSize == 16)) pick = (int)j;      // prefer 16x16, as llama.cpp does
    if (pick < 0) { printf(",\"unsupported\":true,\"why\":\"no matching cooperative-matrix shape\"}\n"); free(p); return 0; }
    uint32_t lm = p[pick].MSize, ln = p[pick].NSize, lk = p[pick].KSize; free(p);
    // round the problem up to whole 2x2 tile blocks; the rate counts the requested shape's operations
    uint32_t M = (m + 2 * lm - 1) / (2 * lm) * (2 * lm), N = (n + 2 * ln - 1) / (2 * ln) * (2 * ln), K = (k + lk - 1) / lk * lk;
    Buf a = make_buf(c, (VkDeviceSize)M * K * es), b = make_buf(c, (VkDeviceSize)K * N * es), cbuf = make_buf(c, (VkDeviceSize)M * N * cs);
    fill(c, &a, fillmode, 1); fill(c, &b, fillmode, 2);
    bind(c, &a, &b, &cbuf);
    uint32_t spec[4] = {lm, ln, lk, c->subgroup};
    VkPipeline pp = pipeline(c, spv, spv_bytes, spec, 4);
    printf(",\"tile\":\"%ux%ux%u\",\"subgroup\":%u", lm, ln, lk, c->subgroup);
    if (!pp) { printf(",\"unsupported\":true,\"why\":\"pipeline creation failed\"}\n"); return 0; }
    uint32_t push[4] = {M, N, K, 0}, gx = M / (2 * lm), gy = N / (2 * ln);
    run(c, pp, gx, gy, push, 1, 0, NULL);
    double* t = calloc(reps, sizeof(double));
    for (int i = 0; i < reps; ++i) t[i] = run(c, pp, gx, gy, push, 1, 0, NULL);
    double ops = 2.0 * m * (double)n * k, med = median(t, reps);
    printf(",\"padded\":\"%ux%ux%u\",\"reps\":%d,\"ops\":%.0f,\"t_median_s\":%.9g,\"t_min_s\":%.9g,\"ops_per_s\":%.6g}\n",
           M, N, K, reps, ops, med, t[0], ops / med);
    return 0;
}

// ---- integrated GPU: shared RAM bandwidth -----------------------------------------------------------------------------
typedef struct { const uint64_t* p; size_t n; atomic_int* stop; double bytes; uint64_t sink; } CpuReader;
static void* cpu_read(void* v) {
    CpuReader* r = v; uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    while (!atomic_load(r->stop)) {
        for (size_t i = 0; i + 4 <= r->n; i += 4) { s0 += r->p[i]; s1 += r->p[i + 1]; s2 += r->p[i + 2]; s3 += r->p[i + 3]; }
        r->bytes += (double)r->n * 8;
    }
    r->sink = s0 + s1 + s2 + s3;
    return NULL;
}

static int shared_mode(Ctx* c, int argc, char** argv) {
    double seconds = atof(arg(argc, argv, "--seconds", "8"));
    int threads = atoi(arg(argc, argv, "--threads", "0"));
    if (threads <= 0) threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    size_t cpu_bytes = (size_t)1 << 30;                                    // 1 GiB per side: far beyond any cache
    uint64_t* mem = aligned_alloc(4096, cpu_bytes);
    for (size_t i = 0; i < cpu_bytes / 8; ++i) mem[i] = i * 0x9e3779b97f4a7c15ULL;
    uint64_t gbytes = (uint64_t)1 << 30;
    if (gbytes > c->props.limits.maxStorageBufferRange) gbytes = c->props.limits.maxStorageBufferRange & ~(uint64_t)0xfffff;
    Buf x = make_buf(c, gbytes), y = make_buf(c, 4096), o = make_buf(c, 4096);
    fill(c, &x, 0, 3);
    bind(c, &x, &y, &o);
    uint32_t op = 0; VkPipeline p = pipeline(c, SPV_STREAM, sizeof(SPV_STREAM), &op, 1);
    uint32_t n = (uint32_t)(gbytes / 16), push[4] = {n, 0, 0, 0}, g = groups_for(n, 256 * 4);
    run(c, p, g, 1, push, 1, 0, NULL);
    const char* phases[] = {"cpu", "gpu", "both"};
    for (int ph = 0; ph < 3; ++ph) {
        atomic_int stop = 0;
        CpuReader* rd = calloc(threads, sizeof(*rd)); pthread_t* th = calloc(threads, sizeof(*th));
        size_t per = cpu_bytes / 8 / threads;
        double t0 = now();
        if (ph != 1) for (int i = 0; i < threads; ++i) { rd[i] = (CpuReader){mem + i * per, per, &stop, 0, 0}; pthread_create(&th[i], NULL, cpu_read, &rd[i]); }
        double gpu_bytes = 0, gpu_time = 0;
        while (now() - t0 < seconds) {
            if (ph == 0) { usleep(20000); continue; }
            double w; run(c, p, g, 1, push, 4, 0, &w); gpu_bytes += 4.0 * gbytes; gpu_time += w;
        }
        atomic_store(&stop, 1);
        double cpu_total = 0;
        if (ph != 1) for (int i = 0; i < threads; ++i) { pthread_join(th[i], NULL); cpu_total += rd[i].bytes; }
        double wall = now() - t0;
        head(c, "shared");
        printf(",\"phase\":\"%s\",\"threads\":%d,\"seconds\":%.3f,\"cpu_bytes_per_s\":%.6g,\"gpu_bytes_per_s\":%.6g}\n",
               phases[ph], ph == 1 ? 0 : threads, wall, cpu_total / wall, gpu_time > 0 ? gpu_bytes / gpu_time : 0.0);
        fflush(stdout);
        free(rd); free(th);
    }
    free(mem);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: vkprobe list|stream|launch|fma|gemm|shared ...\n"); return 1; }
    if (!strcmp(argv[1], "list")) return list_mode();
    Ctx c; open_device(&c, atoi(arg(argc, argv, "--device", "0")));
    if (!strcmp(argv[1], "stream")) return stream_mode(&c, argc, argv);
    if (!strcmp(argv[1], "launch")) return launch_mode(&c, argc, argv);
    if (!strcmp(argv[1], "fma")) return fma_mode(&c, argc, argv);
    if (!strcmp(argv[1], "gemm")) return gemm_mode(&c, argc, argv);
    if (!strcmp(argv[1], "shared")) return shared_mode(&c, argc, argv);
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 1;
}
