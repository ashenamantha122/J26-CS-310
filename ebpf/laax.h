/* Shared definitions between the light and deep BPF programs and user space. */
#ifndef __LAAX_H
#define __LAAX_H

#define TASK_COMM_LEN 16
#define MAX_ARGV_LEN  256

enum laax_kind {
    LAAX_EXEC    = 0,
    LAAX_CONNECT = 1,
    LAAX_ACCEPT  = 2,
};

enum laax_depth {
    LAAX_LIGHT = 0,
    LAAX_DEEP  = 1,
};

/* Mirrors laax.contracts.HostEvent. Keep the two in step. */
struct laax_event {
    __u64 ts_ns;
    __u32 pid;
    __u32 ppid;
    __u8  kind;
    __u8  depth;
    __u16 dport;
    __u32 daddr;
    __u32 saddr;
    char  comm[TASK_COMM_LEN];
    char  pcomm[TASK_COMM_LEN];   /* parent comm, for the lineage rarity key */
    /* Deep mode only. Left zeroed in light mode -- this is the payload whose
       collection cost the adaptive design exists to avoid paying continuously. */
    char  argv[MAX_ARGV_LEN];
    __s32 ret;
};

#endif /* __LAAX_H */
