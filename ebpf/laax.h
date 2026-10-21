/* Shared definitions between the light and deep BPF programs and user space. */
#ifndef __LAAX_H
#define __LAAX_H

#define TASK_COMM_LEN 16
#define MAX_ARGV_LEN  256
#define MAX_PATH_LEN  128

enum laax_kind {
    LAAX_EXEC    = 0,
    LAAX_CONNECT = 1,
    LAAX_ACCEPT  = 2,
    LAAX_FORK    = 3,   /* only emitted when the child inherited deep or deny */
    LAAX_EXIT    = 4,   /* process (thread group) exit */
    LAAX_FILE    = 5,   /* deep only: file open */
};

enum laax_depth {
    LAAX_LIGHT = 0,
    LAAX_DEEP  = 1,
};

/* laax_event.flags on LAAX_FORK: what the child inherited in-kernel. */
#define LAAX_F_INHERIT_DEEP  0x1
#define LAAX_F_INHERIT_DENY  0x2

/* laax_event.family */
#define LAAX_AF_INET   2
#define LAAX_AF_INET6 10

/* Mirrors laax.contracts.HostEvent. Keep the two in step. */
struct laax_event {
    __u64 ts_ns;
    __u32 pid;
    __u32 ppid;
    __u8  kind;
    __u8  depth;
    __u8  family;                 /* AF_INET / AF_INET6 on network events */
    __u8  proto;                  /* IPPROTO_TCP / IPPROTO_UDP, 0 if unknown */
    __u16 dport;
    __u16 sport;
    __u16 argv_len;               /* bytes of argv in use; NUL-separated */
    __u16 _pad;
    __u32 daddr;                  /* IPv4, network order */
    __u32 saddr;
    __u8  daddr6[16];             /* IPv6 */
    __u8  saddr6[16];
    __u32 flags;
    __s32 ret;
    char  comm[TASK_COMM_LEN];
    char  pcomm[TASK_COMM_LEN];   /* parent comm, for the lineage rarity key */
    char  gpcomm[TASK_COMM_LEN];  /* grandparent comm, for v2 ancestry */
    /* exec (light and deep): path of the binary actually executed. comm is
       chosen by the process and can be anything; this is what the kernel
       loaded, so the scorer keys executable-path signals on it. */
    char  filename[MAX_PATH_LEN];
    /* Deep mode only. Left empty in light mode -- this is the payload whose
       collection cost the adaptive design exists to avoid paying continuously.
       exec: argv strings, NUL-separated. file: the opened file's name. */
    char  argv[MAX_ARGV_LEN];
};

#endif /* __LAAX_H */
