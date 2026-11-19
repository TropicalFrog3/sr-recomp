#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_pool_index.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_mem_internal.h"
#include "platform/platform.h"

// ---------------------------------------------------------------- VRAM mirror of buffer-bound pools
// An imported pool keeps vertices, indices, constants
// and storage buffers in system memory, so a discrete GPU reads them over PCIe on every draw (GTX 1050 Ti and RTX
// 3090 Ti: about 110 ms of GPU time per frame in the field). In mirror mode (device class discrete, MKW_MEMORY) an
// imported pool gets a device-local buffer over its imported range the first time the replay uses it as a vertex,
// index, uniform, storage or indirect buffer (vk_mem_use). From then on those bindings, and the buffer copies, fills
// and uniform updates that touch the pool (vk_mem_pick), use the mirror at the same offsets. Guest memory stays what
// the CPU reads and writes; textures and every host path that reads or writes pool bytes keep using it.
// - CPU -> mirror: the first access of a submission to a mirror syncs it (mirror_sync): the 2 MiB write-watch
//   snapshots of its regions for this epoch (snap_regions, shared with the texture checks: the OS bits keep one
//   consumer, the shared page generations) and every page stamped after the mirror's mgen goes into a copy; so do the
//   nvnMemoryPoolFlushMappedRange ranges. The first use copies the whole range. The copies run in a prologue command
//   buffer that vk_sync.c submits right before the part that accessed the mirror first, between full barriers.
//   vk_mem_mirror_begin snaps the regions of the pools the previous submission accessed in a few merged queries.
// - GPU -> guest: what the replay writes into a mirror (storage buffers its shaders write, buffer copies, fills,
//   uniform updates, texture-to-buffer copies: vk_mem_gpu_wrote) goes back into the imported buffer at the end of the
//   submission, exact byte ranges, before the fence signals (vk_mem_mirror_end).
// - MKW_MEMORY=verify: every submission also copies up to VERIFY_BYTES of mirror pages, round robin, into host memory;
//   at its fence the pages the CPU has not written since their sync are compared with guest memory.
// A pool stays imported when its mirror would not fit the VRAM budget, its range is not write-watched, it overlaps
// another pool or the allocation fails.
#define VERIFY_BYTES (4u << 20)
#define VERIFY_SAMPLES 64
#define VERIFY_LOG 16
enum { MR_BUDGET, MR_UNTRACKED, MR_OVERLAP, MR_ALLOC, MR_N };

int vk_mirror_on;
int mirror_verify;
int mirror_gap = 2;             // pre-snap: hot pools at most this many 2 MiB regions apart share one query
static PoolRec **mlist;                // pool_lock: mirrored pools by import_lo
static int nm, mcap;
static uint32_t msub;                  // replay: number of the submission being recorded
static uint32_t mepoch;                // replay: its write-watch snapshot epoch (vk_texture_epoch)
typedef struct { PoolRec *r; uint64_t off, size; } MRange;
static MRange *pend, *wbl;             // replay: copies for the next prologue; GPU writes into mirrors this submission
static int npend, cpend, nwbl, cwbl;
static VkBufferCopy *mregions;         // replay: vkCmdCopyBuffer regions
static int cregions;
static PlatMutex flush_lock = PLAT_MUTEX_INIT;

typedef struct {
    uint64_t subs, syncs, upload_bytes, upload_copies, first_n, first_bytes, wb_bytes, wb_copies, wb_ranges;
    uint64_t presnap_queries, sync_queries, presnap_t, sync_t, prologues, presnap_bytes;
} MirrorAcc;
static MirrorAcc mrep;                 // replay: this submission
static MirrorAcc macc;                 // mirror_stat_lock: since the last summary
static PlatMutex mirror_stat_lock = PLAT_MUTEX_INIT;
volatile int64_t m_live_n, m_live_bytes;
static volatile int64_t m_refused_n[MR_N], m_refused_bytes[MR_N];
static volatile int64_t m_flush_calls, m_flush_bytes, v_pages, v_skipped, v_mismatch;


static void range_add(MRange **a, int *n, int *cap, PoolRec *r, uint64_t off, uint64_t size) {
    if (*n && (*a)[*n - 1].r == r && (*a)[*n - 1].off + (*a)[*n - 1].size == off) {
        (*a)[*n - 1].size += size;
        return;
    }
    if (*n == *cap) vk_mem_grow((void **)a, cap, sizeof **a);
    (*a)[(*n)++] = (MRange){r, off, size};
}

static int range_cmp(const void *x, const void *y) {
    const MRange *a = x, *b = y;
    if (a->r != b->r) return (uintptr_t)a->r < (uintptr_t)b->r ? -1 : 1;
    return a->off < b->off ? -1 : a->off > b->off;
}

// Records the ranges as one vkCmdCopyBuffer per pool, overlapping and adjacent ranges merged; to_mirror: buf -> mbuf,
// else mbuf -> buf. Returns the commands recorded; adds what they move to *bytes and *copies.
static int copy_ranges(VkCommandBuffer cb, MRange *a, int n, int to_mirror, uint64_t *bytes, uint64_t *copies) {
    qsort(a, (size_t)n, sizeof *a, range_cmp);
    int rec = 0;
    for (int i = 0; i < n;) {
        PoolRec *r = a[i].r;
        uint64_t lim = r->import_hi - r->import_lo;
        int k = 0;
        for (; i < n && a[i].r == r; i++) {
            uint64_t lo = a[i].off, hi = a[i].off + a[i].size;
            if (hi > lim) hi = lim;
            if (lo >= hi) continue;
            if (k && lo <= mregions[k - 1].srcOffset + mregions[k - 1].size) {
                uint64_t end = mregions[k - 1].srcOffset + mregions[k - 1].size;
                if (hi > end) mregions[k - 1].size += hi - end;
                continue;
            }
            if (k == cregions) vk_mem_grow((void **)&mregions, &cregions, sizeof *mregions);
            mregions[k++] = (VkBufferCopy){lo, lo, hi - lo};
        }
        if (!k) continue;
        if (to_mirror) vkCmdCopyBuffer(cb, r->buf, r->mbuf, (uint32_t)k, mregions);
        else vkCmdCopyBuffer(cb, r->mbuf, r->buf, (uint32_t)k, mregions);
        rec++;
        for (int j = 0; j < k; j++) *bytes += mregions[j].size;
        *copies += (uint64_t)k;
    }
    return rec;
}

static void mbarrier(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                     VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                           .dstStageMask = ds, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void refuse(PoolRec *r, int why, const char *what) {
    r->mstate = MS_REFUSED;
    uint64_t n = r->import_hi - r->import_lo;
    if (plat_inc64(&m_refused_n[why]) <= 4) {
        char f[160];
        flag_names(r->flags, f, sizeof f);
        rt_log("mem: pool %#llx+%#llx (%s) stays imported: %s", (unsigned long long)r->lo, (unsigned long long)r->size,
               f, what);
    }
    plat_add64(&m_refused_bytes[why], (int64_t)n);
}

// Replay thread: the pool's mirror on its first buffer use.
static void mirror_create(PoolRec *r) {
    VkCtx *c = vk_ctx();
    uint64_t n = r->import_hi - r->import_lo;
    r->mstate = MS_REFUSED;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    if (r->kind != POOL_IMPORT || !r->buf || !n) return;
    plat_mutex_lock_shared(&pool_lock);
    int i = pidx_upper(&pool_index, r->lo) - 1;
    while (i >= 0 && pool_index.e[i].buf != r) i--;
    int alone = i >= 0 && (i == 0 || pool_index.maxhi[i - 1] <= pool_index.e[i].lo) &&
                (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= pool_index.e[i].hi);
    plat_mutex_unlock_shared(&pool_lock);
    if (i < 0) return;   // finalized meanwhile
    if (!alone) return refuse(r, MR_OVERLAP, "it overlaps another pool");
    uint64_t r0 = r->import_lo >> SNAP_SHIFT, r1 = (r->import_hi - 1) >> SNAP_SHIFT;
    if (r1 >= SNAP_REGIONS) return refuse(r, MR_UNTRACKED, "outside the write-watched address space");
    uint64_t used, budget;
    int exact;
    if (vk_mem_vram(&used, &budget, &exact) && used + n + budget / 8 > budget) {
        char why[128];
        snprintf(why, sizeof why, "its %.1f MiB mirror does not fit the VRAM budget (%.0f of %.0f MiB in use)",
                 n / 1048576.0, used / 1048576.0, budget / 1048576.0);
        return refuse(r, MR_BUDGET, why);
    }
    gen_lock_x();
    mrep.sync_queries += snap_regions(r0, r1, mepoch);
    int untracked = 0;
    for (uint64_t k = r0; k <= r1; k++) untracked |= snap_untracked[k];
    r->mgen = gen_now;   // the whole-range copy below holds every write observed so far
    plat_mutex_unlock(&gen_lock);
    if (untracked) return refuse(r, MR_UNTRACKED, "its range is not write-watched");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    VkResult res = vmaCreateBuffer(c->vma, &bi, &ai, &r->mbuf, &r->malloc_, NULL);
    if (res != VK_SUCCESS) {
        r->mbuf = VK_NULL_HANDLE, r->malloc_ = NULL;
        char why[96];
        snprintf(why, sizeof why, "device-local allocation of %.1f MiB failed (%d)", n / 1048576.0, (int)res);
        return refuse(r, MR_ALLOC, why);
    }
    plat_mutex_lock(&pool_lock);
    if (nm == mcap) vk_mem_grow((void **)&mlist, &mcap, sizeof *mlist);
    int at = nm;
    while (at > 0 && mlist[at - 1]->import_lo > r->import_lo) at--;
    memmove(mlist + at + 1, mlist + at, (size_t)(nm - at) * sizeof *mlist);
    mlist[at] = r;
    nm++;
    plat_mutex_unlock(&pool_lock);
    r->acc_sub = msub;
    __atomic_store_n(&r->mstate, MS_ON, __ATOMIC_RELEASE);   // after mbuf: vk_mem_use reads it without use_lock
    range_add(&pend, &npend, &cpend, r, 0, n);
    mrep.first_n++, mrep.first_bytes += n;
    plat_inc64(&m_live_n);
    plat_add64(&m_live_bytes, (int64_t)n);
}

void mirror_forget(PoolRec *r) {
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    for (int i = 0; i < nm; i++)
        if (mlist[i] == r) {
            memmove(mlist + i, mlist + i + 1, (size_t)(nm - i - 1) * sizeof *mlist);
            nm--;
            plat_dec64(&m_live_n);
            plat_add64(&m_live_bytes, -(int64_t)(r->import_hi - r->import_lo));
            break;
        }
}

// Finalized mirrored pools waiting for mirror_retire; vk_mem_shutdown destroys those whose callback never ran.
static PoolRec **retiring;
static int nretiring, cretiring;
static PlatMutex retire_lock = PLAT_MUTEX_INIT;

void mirror_retiring(PoolRec *r) {
    plat_mutex_lock(&retire_lock);
    if (nretiring == cretiring) vk_mem_grow((void **)&retiring, &cretiring, sizeof *retiring);
    retiring[nretiring++] = r;
    plat_mutex_unlock(&retire_lock);
}

int retiring_count(void) {
    plat_mutex_lock(&retire_lock);
    int n = nretiring;
    plat_mutex_unlock(&retire_lock);
    return n;
}

void mirror_retire(void *arg) {
    plat_mutex_lock(&retire_lock);
    for (int i = 0; i < nretiring; i++)
        if (retiring[i] == arg) { retiring[i] = retiring[--nretiring]; break; }
    plat_mutex_unlock(&retire_lock);
    destroy_rec(vk_ctx(), arg);
}

static void mirror_pull_reset(VkCtx *c);

// vk_mem_shutdown, pool_lock held, after the indexed pools went: mirrors went with their pools; finalized pools whose
// callback never ran go now.
void mirror_reset(VkCtx *c) {
    nm = npend = nwbl = 0;
    for (int i = 0; i < nretiring; i++) destroy_rec(c, retiring[i]);
    nretiring = 0;
    mirror_pull_reset(c);
}

// Replay thread, the first access of this submission: queues the copies of the pages written since the last sync.
static void mirror_sync(PoolRec *r) {
    uint64_t t0 = qpc();
    r->acc_sub = msub;
    uint64_t lo = r->import_lo, hi = r->import_hi;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    int excl = 0;
    gen_lock_s();
    for (uint64_t k = r0; k <= r1; k++)
        if (snap_at[k] != mepoch) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        gen_lock_x();
        mrep.sync_queries += snap_regions(r0, r1, mepoch);
    }
    uint32_t since = r->mgen;
    for (uint64_t k = r0; k <= r1; k++) {
        if (reg_max[k] <= since) continue;
        uint64_t a = k << SNAP_SHIFT, b = a + (1ull << SNAP_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = gen_chunk[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        if (!ch) continue;
        for (uint64_t x = a; x < b; x += 1ull << GEN_PAGE_SHIFT)
            if (ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] > since)
                range_add(&pend, &npend, &cpend, r, x - lo, 1ull << GEN_PAGE_SHIFT);
    }
    r->mgen = gen_now;
    if (excl) plat_mutex_unlock(&gen_lock);
    else plat_mutex_unlock_shared(&gen_lock);
    if (__atomic_load_n(&r->fl_hi, __ATOMIC_ACQUIRE)) {
        plat_mutex_lock(&flush_lock);
        uint64_t a = r->fl_lo, b = r->fl_hi;
        r->fl_hi = 0;
        plat_mutex_unlock(&flush_lock);
        a = a > lo ? a & ~0xfffull : lo;
        b = b < hi ? (b + 0xfff) & ~0xfffull : hi;
        if (a < b) range_add(&pend, &npend, &cpend, r, a - lo, b - a);
    }
    mrep.syncs++;
    mrep.sync_t += qpc() - t0;
}

// The replay's chunks record on several threads (MKW_REPLAY_PAR): a pool's first access of the submission (creation,
// sync) and the write-back list go under use_lock; a pool already synced in this submission is read without it.
static PlatMutex use_lock = PLAT_MUTEX_INIT;

void vk_mem_use(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    if (__atomic_load_n(&r->mstate, __ATOMIC_ACQUIRE) != MS_ON || __atomic_load_n(&r->acc_sub, __ATOMIC_ACQUIRE) != msub) {
        plat_mutex_lock(&use_lock);
        if (r->mstate == MS_NONE) mirror_create(r);
        if (r->mstate == MS_ON && r->acc_sub != msub) mirror_sync(r);
        plat_mutex_unlock(&use_lock);
        if (r->mstate != MS_ON) return;
    }
    sp->buffer = r->mbuf;
}

void vk_mem_pick(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    if (!r || __atomic_load_n(&r->mstate, __ATOMIC_ACQUIRE) != MS_ON) return;
    if (__atomic_load_n(&r->acc_sub, __ATOMIC_ACQUIRE) != msub) {
        plat_mutex_lock(&use_lock);
        if (r->acc_sub != msub) mirror_sync(r);
        plat_mutex_unlock(&use_lock);
    }
    sp->buffer = r->mbuf;
}

int vk_mem_mirror_probe(uint64_t addr) {
    VkMemSpan sp;
    if (!vk_mirror_on || !vk_mem_resolve(addr, 1, &sp) || !sp.pool) return 0;
    mepoch = vk_texture_epoch();
    vk_mem_use(&sp);
    return ((PoolRec *)sp.pool)->mstate == MS_ON;
}

int cap_busy(void);   // cap.h

static MRange *mwbl;
static int nmwbl, cmwbl;
static PlatMutex mwb_lock = PLAT_MUTEX_INIT;    // leaf: mwb_bits, mwb_sub_fence, mwb_active
static PlatMutex pull_lock = PLAT_MUTEX_INIT;   // mwblist and pull command buffer
static PoolRec **mwblist;
static int nmwblist, cap_mwblist, pull_hook_set;
static VkCommandPool pull_cmd_pool;
static VkCommandBuffer pull_cb;
static VkFence pull_fence;
static volatile int64_t m_pull_calls, m_pull_pages, m_pull_bytes;

int vk_mem_mirror_lazy_on(void) {
#ifdef _WIN32
    return 0;
#else
    static int v = -1, rt = -1;
    if (v < 0) {
        const char *e = cfg_get("MKW_LAZY_WB");
        v = !e || !*e || strcmp(e, "0");
        const char *re = cfg_get("MKW_READ_TRAPS");
        rt = !re || !*re || strcmp(re, "0");
    }
    return v && rt && vk_mirror_on && !mirror_verify && !cap_busy();
#endif
}

static void mwb_register(PoolRec *r) {
    if (__atomic_load_n(&r->mwb_active, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&pull_lock);
    if (!r->mwb_active) {
        uint64_t base_pg = r->import_lo & ~4095ull;
        uint64_t end_pg = (r->import_hi + 4095ull) & ~4095ull;
        uint64_t np = (end_pg - base_pg) >> 12;
        uint64_t nw = (np + 63) >> 6;
        if (!r->mwb_bits) {
            r->mwb_bits = calloc(nw ? nw : 1, sizeof *r->mwb_bits);
            if (!r->mwb_bits) rt_fatal("vk: mirror wb bitmask OOM");
            r->mwb_words = nw;
        }
        if (nmwblist == cap_mwblist) {
            int nc = cap_mwblist ? cap_mwblist * 2 : 16;
            PoolRec **nl = realloc(mwblist, (size_t)nc * sizeof *nl);
            if (!nl) rt_fatal("vk: mirror wb list OOM");
            mwblist = nl, cap_mwblist = nc;
        }
        mwblist[nmwblist++] = r;
        if (!pull_hook_set) {
            mem_trap_pull_hook(vk_mem_mirror_pull);
            pull_hook_set = 1;
        }
        __atomic_store_n(&r->mwb_active, 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&pull_lock);
}

static void mwb_set_bits(PoolRec *r, uint64_t off, uint64_t size, uint64_t fence) {
    if (!r || r->mstate != MS_ON || !r->smap || !size) return;
    uint64_t glo = r->import_lo + off;
    uint64_t gend = glo + size;
    if (gend > r->import_hi) gend = r->import_hi;
    uint64_t p0_g = (glo + 4095ull) & ~4095ull;
    uint64_t p1_g = gend & ~4095ull;
    if (p1_g <= p0_g) return;
    mwb_register(r);
    uint64_t base_pg = r->import_lo & ~4095ull;
    uint64_t p0 = (p0_g - base_pg) >> 12;
    uint64_t p1 = (p1_g - base_pg) >> 12;
    plat_mutex_lock(&mwb_lock);
    if (r->mwb_active && r->mwb_bits) {
        uint64_t maxp = r->mwb_words * 64;
        if (p1 > maxp) p1 = maxp;
        if (p1 > p0) {
            uint64_t w0 = p0 >> 6, w1 = (p1 - 1) >> 6;
            uint64_t m0 = ~0ull << (p0 & 63);
            uint64_t m1 = (p1 & 63) ? ((1ull << (p1 & 63)) - 1) : ~0ull;
            if (w0 == w1) r->mwb_bits[w0] |= (m0 & m1);
            else {
                r->mwb_bits[w0] |= m0;
                for (uint64_t w = w0 + 1; w < w1; w++) r->mwb_bits[w] = ~0ull;
                r->mwb_bits[w1] |= m1;
            }
            if (fence > r->mwb_sub_fence) r->mwb_sub_fence = fence;
        }
    }
    plat_mutex_unlock(&mwb_lock);
}

void vk_mem_mirror_note_written(PoolRec *r, uint64_t addr, uint64_t size) {
    if (!r || r->mstate != MS_ON || !r->smap || !size) return;
    if (addr < r->import_lo || addr + size > r->import_hi) return;
    if (!vk_mem_mirror_lazy_on()) return;
    mwb_set_bits(r, addr - r->import_lo, size, 0);
}

void mirror_pull_forget(PoolRec *r) {
    if (!r || (!__atomic_load_n(&r->mwb_active, __ATOMIC_ACQUIRE) && !r->mwb_bits)) return;
    plat_mutex_lock(&pull_lock);
    plat_mutex_lock(&mwb_lock);
    r->mwb_active = 0;
    plat_mutex_unlock(&mwb_lock);
    for (int i = 0; i < nmwblist; i++) {
        if (mwblist[i] == r) {
            mwblist[i] = mwblist[--nmwblist];
            break;
        }
    }
    plat_mutex_unlock(&pull_lock);
}

void vk_mem_gpu_wrote(const VkMemSpan *sp, uint64_t size) {
    PoolRec *r = sp->pool;
    if (!r || r->mstate != MS_ON || sp->buffer != r->mbuf || !size) return;
    uint64_t off = sp->offset, end = off + size;
    if (size > (64u << 10) && r->smap && vk_mem_mirror_lazy_on() && r->import_lo + end <= r->import_hi) {
        uint64_t glo = r->import_lo + off, gend = r->import_lo + end;
        uint64_t p0_g = (glo + 4095ull) & ~4095ull;
        uint64_t p1_g = gend & ~4095ull;
        if (p1_g > p0_g) {
            plat_mutex_lock(&use_lock);
            if (p0_g > glo) range_add(&wbl, &nwbl, &cwbl, r, off, p0_g - glo);
            range_add(&mwbl, &nmwbl, &cmwbl, r, p0_g - r->import_lo, p1_g - p0_g);
            if (gend > p1_g) range_add(&wbl, &nwbl, &cwbl, r, p1_g - r->import_lo, gend - p1_g);
            plat_mutex_unlock(&use_lock);
            return;
        }
    }
    plat_mutex_lock(&use_lock);
    range_add(&wbl, &nwbl, &cwbl, r, off, size);
    plat_mutex_unlock(&use_lock);
}

void nvn_pool_flushed(const pool_t *p, uint64_t off, uint64_t size) {
    if (!vk_mirror_on || !size || !p->mem) return;
    plat_inc64(&m_flush_calls);
    plat_add64(&m_flush_bytes, (int64_t)size);
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)p->mem, 0);
    PoolRec *r = e && e->key == p ? e->buf : NULL;
    if (r && r->mstate == MS_ON) {
        uint64_t a = (uintptr_t)p->mem + off, b = a + size;
        plat_mutex_lock(&flush_lock);
        if (!r->fl_hi) r->fl_lo = a, r->fl_hi = b;
        else {
            if (a < r->fl_lo) r->fl_lo = a;
            if (b > r->fl_hi) r->fl_hi = b;
        }
        plat_mutex_unlock(&flush_lock);
    }
    plat_mutex_unlock_shared(&pool_lock);
}


void vk_mem_mirror_begin(void) {
    if (!vk_mirror_on) {   // import mode: the texture checks' snapshots still skip frozen regions
        if (!freeze_idle) return;
        gen_lock_x();
        ww_ep++;
        freeze_pass();
        plat_mutex_unlock(&gen_lock);
        return;
    }
    uint64_t t0 = qpc();
    uint32_t prev = msub;
    msub = msub + 1 ? msub + 1 : 1;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    mepoch = vk_texture_epoch();
    uint64_t q = 0, run0 = 0, run1 = 0;
    int have = 0;
    plat_mutex_lock_shared(&pool_lock);
    gen_lock_x();
    ww_ep++;
    uint64_t b0 = ww_bytes;
    for (int i = 0; i < nm; i++) {
        const PoolRec *r = mlist[i];
        if (r->acc_sub != prev) continue;
        uint64_t a = r->import_lo >> SNAP_SHIFT, b = (r->import_hi - 1) >> SNAP_SHIFT;
        if (have && a <= run1 + 1 + (uint64_t)mirror_gap) {
            if (b > run1) run1 = b;
            continue;
        }
        if (have) q += snap_regions(run0, run1, mepoch);
        run0 = a, run1 = b, have = 1;
    }
    if (have) q += snap_regions(run0, run1, mepoch);
    mrep.presnap_bytes += ww_bytes - b0;
    if (freeze_idle) freeze_pass();
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock_shared(&pool_lock);
    mrep.subs++;
    mrep.presnap_queries += q;
    mrep.presnap_t += qpc() - t0;
}

int vk_mem_mirror_pending(void) { return npend; }

void vk_mem_mirror_prologue(VkCommandBuffer cb) {
    if (!npend) return;
    // #43 (RX 6650 XT, a discrete card: mirror mode) hung after every marked command had finished
    uint32_t crumb = vk_crumb_op(cb, "mirror copies at the submission's start", (uint32_t)npend, 0);
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    copy_ranges(cb, pend, npend, 1, &mrep.upload_bytes, &mrep.upload_copies);
    mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    vk_crumb_end(cb, VK_CRUMB_OP, crumb);
    npend = 0;
    mrep.prologues++;
}

// ---- verify mode
typedef struct { PoolRec *r; uint64_t off, len, dst; uint32_t mgen; } VSample;
typedef struct { uint32_t sub; int n; VkBuffer buf; VmaAllocation alloc; const uint8_t *map; VSample s[VERIFY_SAMPLES]; } VJob;
static int ver_i;
static uint64_t ver_off;

// The CPU wrote the page since the mirror's sync that mgen records: stamped later, or its write-watch bit is set.
static int page_dirty(uint64_t a, uint32_t mgen) {
    uint32_t *s = gen_slot(a, 0);
    if (s && __atomic_load_n(s, __ATOMIC_RELAXED) > mgen) return 1;
    uint64_t p[1];
    return mem_written_peek(a, 1ull << GEN_PAGE_SHIFT, p, 1) != 0;   // UINT64_MAX (not watched) counts as written
}

static void verify_cb(void *arg) {   // completion thread
    VJob *j = arg;
    static volatile int32_t logged;
    uint64_t pages = 0, skipped = 0, bad = 0;
    for (int i = 0; i < j->n; i++) {
        const VSample *s = &j->s[i];
        PoolRec *r = s->r;
        uint32_t wb = __atomic_load_n(&r->wb_sub, __ATOMIC_RELAXED);
        if (wb != j->sub && wb - j->sub < 0x80000000u) {   // a later submission wrote back into it
            skipped += s->len >> GEN_PAGE_SHIFT;
            continue;
        }
        for (uint64_t o = 0; o < s->len; o += 1ull << GEN_PAGE_SHIFT) {
            uint64_t a = r->import_lo + s->off + o;
            const uint8_t *m = j->map + s->dst + o, *g = (const uint8_t *)(uintptr_t)a;
            pages++;
            if (!memcmp(m, g, 1ull << GEN_PAGE_SHIFT)) continue;
            if (page_dirty(a, s->mgen)) { skipped++; continue; }
            bad++;
            if (plat_inc32(&logged) <= VERIFY_LOG) {
                int k = 0, e = (1 << GEN_PAGE_SHIFT) - 1;
                while (k < e && m[k] == g[k]) k++;
                while (e > k && m[e] == g[e]) e--;
                char f[160];
                flag_names(r->flags, f, sizeof f);
                rt_log("mem: verify mismatch: pool %#llx+%#llx (%s) offset %#llx: bytes %#x..%#x of the page differ, "
                       "mirror %02x guest %02x at the first (submission %u)", (unsigned long long)r->lo,
                       (unsigned long long)r->size, f, (unsigned long long)(a - r->lo), k, e, m[k], g[k], j->sub);
            }
        }
    }
    plat_add64(&v_pages, (int64_t)pages);
    plat_add64(&v_skipped, (int64_t)skipped);
    plat_add64(&v_mismatch, (int64_t)bad);
    vmaDestroyBuffer(vk_ctx()->vma, j->buf, j->alloc);
    free(j);
}

static int verify_record(VkCommandBuffer cb, uint64_t fence) {
    VJob *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("vk: out of memory");
    uint64_t left = VERIFY_BYTES, dst = 0;
    plat_mutex_lock_shared(&pool_lock);
    for (int tries = 0; nm && left && j->n < VERIFY_SAMPLES && tries <= nm; tries++) {
        if (ver_i >= nm) ver_i = 0, ver_off = 0;
        PoolRec *r = mlist[ver_i];
        uint64_t size = r->import_hi - r->import_lo;
        if (ver_off >= size) { ver_i++, ver_off = 0; continue; }
        uint64_t len = size - ver_off < left ? size - ver_off : left;
        j->s[j->n++] = (VSample){r, ver_off, len, dst, r->mgen};
        dst += len, left -= len, ver_off += len;
        if (ver_off >= size) ver_i++, ver_off = 0;
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (!j->n) { free(j); return 0; }
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = dst,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info) != VK_SUCCESS) { free(j); return 0; }
    j->map = info.pMappedData;
    j->sub = msub;
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < j->n; i++) {
        VkBufferCopy c = {j->s[i].off, j->s[i].dst, j->s[i].len};
        vkCmdCopyBuffer(cb, j->s[i].r->mbuf, j->buf, 1, &c);
    }
    mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
             VK_ACCESS_2_HOST_READ_BIT);
    vk_completion_add(fence, verify_cb, j);
    return j->n + 2;
}

int vk_mem_mirror_flush_part(VkCommandBuffer cb) {
    if (!vk_mirror_on || !nwbl) return 0;
    int rec = 0;
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < nwbl; i++) wbl[i].r->wb_sub = msub;
    mrep.wb_ranges += (uint64_t)nwbl;
    rec += copy_ranges(cb, wbl, nwbl, 0, &mrep.wb_bytes, &mrep.wb_copies) + 2;
    mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
             VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    nwbl = 0;
    return rec;
}

int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence) {
    if (!vk_mirror_on) return 0;
    int rec = 0;
    if (nwbl) {
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
        for (int i = 0; i < nwbl; i++) wbl[i].r->wb_sub = msub;
        mrep.wb_ranges += (uint64_t)nwbl;
        rec += copy_ranges(cb, wbl, nwbl, 0, &mrep.wb_bytes, &mrep.wb_copies) + 2;
        mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        nwbl = 0;
    }
    if (nmwbl) {
        for (int i = 0; i < nmwbl; i++) {
            mwbl[i].r->wb_sub = msub;
            mwb_set_bits(mwbl[i].r, mwbl[i].off, mwbl[i].size, fence);
        }
        nmwbl = 0;
    }
    if (mirror_verify) rec += verify_record(cb, fence);
    plat_mutex_lock(&mirror_stat_lock);
    uint64_t *d = (uint64_t *)&macc, *s = (uint64_t *)&mrep;
    for (size_t i = 0; i < sizeof macc / sizeof(uint64_t); i++) d[i] += s[i];
    plat_mutex_unlock(&mirror_stat_lock);
    memset(&mrep, 0, sizeof mrep);
    return rec;
}

void vk_mem_mirror_pull(uint64_t addr, uint64_t size) {
    if (!size) return;
    uint64_t req_lo = addr & ~4095ull;
    uint64_t req_hi = (addr + size + 4095ull) & ~4095ull;
    plat_mutex_lock(&pull_lock);
    VkCtx *c = vk_ctx();
    if (!c) { plat_mutex_unlock(&pull_lock); return; }
    for (int mi = 0; mi < nmwblist; mi++) {
        PoolRec *r = mwblist[mi];
        if (!r || r->import_hi <= req_lo || r->import_lo >= req_hi) continue;
        uint64_t base_pg = r->import_lo & ~4095ull;
        uint64_t plo = req_lo > base_pg ? req_lo : base_pg;
        uint64_t phi = req_hi < (r->import_hi & ~4095ull) ? req_hi : (r->import_hi & ~4095ull);
        if (phi <= plo) continue;
        uint64_t p0 = (plo - base_pg) >> 12;
        uint64_t p1 = (phi - base_pg) >> 12;

        plat_mutex_lock(&mwb_lock);
        if (!r->mwb_active || !r->mwb_bits) { plat_mutex_unlock(&mwb_lock); continue; }
        uint64_t maxp = r->mwb_words * 64;
        if (p1 > maxp) p1 = maxp;
        if (p1 <= p0) { plat_mutex_unlock(&mwb_lock); continue; }
        int any = 0;
        for (uint64_t w = p0 >> 6, we = (p1 - 1) >> 6; w <= we; w++) {
            uint64_t m = r->mwb_bits[w];
            if (w == (p0 >> 6)) m &= ~0ull << (p0 & 63);
            if (w == we && (p1 & 63)) m &= (1ull << (p1 & 63)) - 1;
            if (m) { any = 1; break; }
        }
        if (!any) { plat_mutex_unlock(&mwb_lock); continue; }

        uint64_t e0 = p0, e1 = p1;
        if (p1 - p0 == 1) {
            uint64_t pool_pages = ((r->import_hi & ~4095ull) - base_pg) >> 12;
            if (pool_pages > maxp) pool_pages = maxp;
            while (e0 > 0 && (p1 - e0) < 128) {
                uint64_t prev = e0 - 1;
                if (!((r->mwb_bits[prev >> 6] >> (prev & 63)) & 1)) break;
                if (!mem_trapped(base_pg + (prev << 12))) break;
                e0 = prev;
            }
            while (e1 < pool_pages && (e1 - e0) < 256) {
                if (!((r->mwb_bits[e1 >> 6] >> (e1 & 63)) & 1)) break;
                if (!mem_trapped(base_pg + (e1 << 12))) break;
                e1++;
            }
        }

        VkBufferCopy stack_regs[64];
        VkBufferCopy *regs = stack_regs;
        int cap_r = 64, nreg = 0;
        uint64_t copied_pages = 0, copied_bytes = 0;
        for (uint64_t p = e0; p < e1;) {
            if (!((r->mwb_bits[p >> 6] >> (p & 63)) & 1)) { p++; continue; }
            uint64_t s = p;
            while (p < e1 && ((r->mwb_bits[p >> 6] >> (p & 63)) & 1)) {
                r->mwb_bits[p >> 6] &= ~(1ull << (p & 63));
                p++;
            }
            uint64_t ga = base_pg + (s << 12), gb = base_pg + (p << 12);
            if (ga < r->import_lo) ga = r->import_lo;
            if (gb > r->import_hi) gb = r->import_hi;
            if (gb <= ga) continue;
            if (nreg == cap_r) {
                int nc = cap_r * 2;
                VkBufferCopy *nr = regs == stack_regs ? malloc((size_t)nc * sizeof *nr) : realloc(regs, (size_t)nc * sizeof *nr);
                if (!nr) rt_fatal("vk: mirror pull regions OOM");
                if (regs == stack_regs) memcpy(nr, stack_regs, (size_t)nreg * sizeof *nr);
                regs = nr, cap_r = nc;
            }
            uint64_t off = ga - r->import_lo, sz = gb - ga;
            regs[nreg++] = (VkBufferCopy){.srcOffset = off, .dstOffset = off, .size = sz};
            copied_pages += (p - s);
            copied_bytes += sz;
        }
        plat_mutex_unlock(&mwb_lock);

        if (nreg) {
            if (!pull_cmd_pool) {
                VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                              .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                              .queueFamilyIndex = c->gfx_family};
                VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                  .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
                VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                if (vkCreateCommandPool(c->device, &pi, NULL, &pull_cmd_pool) != VK_SUCCESS)
                    rt_fatal("vk: mirror pull pool");
                ai.commandPool = pull_cmd_pool;
                if (vkAllocateCommandBuffers(c->device, &ai, &pull_cb) != VK_SUCCESS ||
                    vkCreateFence(c->device, &fi, NULL, &pull_fence) != VK_SUCCESS)
                    rt_fatal("vk: mirror pull cb/fence");
            }
            vkResetFences(c->device, 1, &pull_fence);
            vkResetCommandBuffer(pull_cb, 0);
            VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                           .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            vkBeginCommandBuffer(pull_cb, &bi);
            mbarrier(pull_cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            vkCmdCopyBuffer(pull_cb, r->mbuf, r->buf, (uint32_t)nreg, regs);
            mbarrier(pull_cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            vkEndCommandBuffer(pull_cb);
            VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &pull_cb};
            vk_queue_lock();
            vk_sync_parts_flush();
            VkSparseOrder so = vk_sparse_order(0, 0);
            VkPipelineStageFlags ws = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkTimelineSemaphoreSubmitInfo tl = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                                .waitSemaphoreValueCount = so.wait ? 1u : 0u, .pWaitSemaphoreValues = &so.wait,
                                                .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &so.signal};
            if (so.ssem) si.pNext = &tl, si.signalSemaphoreCount = 1, si.pSignalSemaphores = &so.ssem;
            if (so.ssem && so.wait) si.waitSemaphoreCount = 1, si.pWaitSemaphores = &so.wsem, si.pWaitDstStageMask = &ws;
            VkResult sr = vkQueueSubmit(c->gfx_queue, 1, &si, pull_fence);
            vk_queue_unlock();
            if (sr != VK_SUCCESS) rt_fatal("vk: mirror pull submit (%d)", (int)sr);
            vkWaitForFences(c->device, 1, &pull_fence, VK_TRUE, UINT64_MAX);
            if (r->salloc) {
                for (int i = 0; i < nreg; i++)
                    vmaInvalidateAllocation(c->vma, r->salloc, regs[i].dstOffset, regs[i].size);
            }
            plat_add64(&m_pull_calls, 1);
            plat_add64(&m_pull_pages, (int64_t)copied_pages);
            plat_add64(&m_pull_bytes, (int64_t)copied_bytes);
        }
        if (regs != stack_regs) free(regs);
    }
    plat_mutex_unlock(&pull_lock);
}

static void mirror_pull_reset(VkCtx *c) {
    mem_trap_pull_hook(NULL);
    plat_mutex_lock(&pull_lock);
    if (c && pull_fence) vkDestroyFence(c->device, pull_fence, NULL);
    if (c && pull_cmd_pool) vkDestroyCommandPool(c->device, pull_cmd_pool, NULL);
    pull_fence = VK_NULL_HANDLE;
    pull_cmd_pool = VK_NULL_HANDLE;
    pull_cb = VK_NULL_HANDLE;
    free(mwblist);
    mwblist = NULL;
    nmwblist = cap_mwblist = 0;
    pull_hook_set = 0;
    free(mwbl);
    mwbl = NULL;
    nmwbl = cmwbl = 0;
    plat_mutex_unlock(&pull_lock);
}

// One summary line: with MKW_TRACE=perf every 60 frames, else at frames 600 and 1800.
void mirror_log(uint64_t frame) {
    static uint64_t last;
    ww_log(frame);
    if (!vk_mirror_on) return;
    plat_mutex_lock(&mirror_stat_lock);
    MirrorAcc a = macc;
    memset(&macc, 0, sizeof macc);
    plat_mutex_unlock(&mirror_stat_lock);
    double f = frame > last ? (double)(frame - last) : 1.0, M = 1048576.0;
    last = frame;
    uint64_t fc = (uint64_t)plat_xchg64(&m_flush_calls, 0), fb = (uint64_t)plat_xchg64(&m_flush_bytes, 0);
    int64_t pc = plat_xchg64(&m_pull_calls, 0), pb = plat_xchg64(&m_pull_bytes, 0);
    plat_xchg64(&m_pull_pages, 0);
    char ver[160] = "";
    if (mirror_verify)
        snprintf(ver, sizeof ver, "; verify: pages compared %lld, skipped %lld (written since), mismatches %lld",
                 (long long)plat_xchg64(&v_pages, 0), (long long)plat_xchg64(&v_skipped, 0),
                 (long long)plat_xchg64(&v_mismatch, 0));
    rt_log("mem: mirror frame %llu: pools %lld, %.1f MiB device-local; stay imported: budget %lld (%.1f MiB), untracked "
           "%lld, overlapping %lld, failed %lld; per frame: submits %.2f, syncs %.1f, upload %.3f MiB in %.1f ranges, "
           "first use %.2f pools %.3f MiB, write-back %.3f MiB in %.1f ranges (lazy pulls %lld %.3f MiB), flush calls %.1f (%.3f MiB), queries "
           "%.1f pre-snap (%.1f MiB) + %.1f sync, prologues %.2f, tracking cpu %.3f ms (pre-snap %.3f, syncs %.3f)%s",
           (unsigned long long)frame, (long long)m_live_n, m_live_bytes / M, (long long)m_refused_n[MR_BUDGET],
           m_refused_bytes[MR_BUDGET] / M, (long long)m_refused_n[MR_UNTRACKED], (long long)m_refused_n[MR_OVERLAP],
           (long long)m_refused_n[MR_ALLOC], a.subs / f, a.syncs / f, a.upload_bytes / f / M, a.upload_copies / f,
           a.first_n / f, a.first_bytes / f / M, a.wb_bytes / f / M, a.wb_copies / f, (long long)pc, pb / M, fc / f, fb / f / M,
           a.presnap_queries / f, a.presnap_bytes / f / M, a.sync_queries / f, a.prologues / f, qpc_ms(a.presnap_t + a.sync_t) / f,
           qpc_ms(a.presnap_t) / f, qpc_ms(a.sync_t) / f, ver);
}

