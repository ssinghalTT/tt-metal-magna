#include "dev_mem_map.h"
#include "memory.ld"

#define FIRMWARE_STACK_SIZE LENGTH(REGION_STACK)
#if defined(COMPILE_FOR_NCRISC)
#define KERNEL_ENTRY_SYMBOL _Z11kernel_initm
#endif

#include "sections.ld"
