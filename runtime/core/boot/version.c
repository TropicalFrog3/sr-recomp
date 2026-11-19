// The rows of version.h. 1.0.0's are the addresses the runtime had before there was a second version; 1.8.0's were
// found in its executable (11/11/2026) by the same features the 1.0.0 comments describe:
// - nvn_ret: main+0xdd07c8 is the first of the two nvnBootstrapLoader calls in the graphics bring-up (main+0xdd0720,
//   after nv::InitializeGraphics and nn::vi::GetDisplayVsyncEvent, as 1.0.0's main+0xcf39b4).
// - frame_update: main+0x69cbb0 is the only function of main using the frame object's +0x210 (fixed interval), +0x214,
//   +0x218, +0x128 and +0xf0 (game.c); vtable column main+0x367a9e0 slot 28, as 1.0.0's; the game clock is the u32
//   it adds [+0x214] to (adrp/ldr/add/str at main+0x69cd30).
// - zstd: recomp/patches-mkw-180.txt says how each function was found.
// - sdk: tools/gpu/nvn2_procs.py locate(), checked by disassembly.
// Not found yet for 1.8.0 (0): the shadow, seed and aspect sets' functions and the menu focus probe.
#include <string.h>

#include "version.h"
#include "host.h"

static const RtVersion versions[] = {
    {
        .number = 100,
        .display = "1.0.0",
        .nvn_ret = 0x31b6ec4,
        .sdk_name_pool = 0xc7105a,
        .sdk_name_offsets = 0xc76478,
        .sdk_pointers = 0xe95bf8,
        .sdk_last_index = 0x2da,
        .inits = {{"main", 5403}, {"subsdk0", 0}, {"sdk", 95}},
    },
};

const RtVersion *rt_version(void) {
    static const RtVersion *cur;
    if (cur) return cur;
    int v = rt_game()->version;
    for (size_t i = 0; i < sizeof versions / sizeof *versions; i++)
        if (versions[i].number == v) return cur = &versions[i];
    rt_fatal("version: the game code is for game version %d, which this runtime has no addresses for", v);
}

uint64_t rt_main_addr(uint64_t off) {
    if (!off) return 0;
    for (int i = 0; i < ldr_count(); i++)
        if (!strcmp(ldr_module(i)->name, "main")) return ldr_module(i)->base + off;
    rt_fatal("version: main is not loaded");
}
