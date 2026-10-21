/*
 * LAAX Component 1 -- kernel side.
 *
 * One object, three tiers of program, attached independently:
 *
 *   LIGHT  static tracepoints (plus one BTF tracepoint for fork). Attached at
 *          start-up, never detached. Stable ABI, cheaper than kprobes, carries
 *          no payload.
 *   DEEP   kprobes/kretprobes. Loaded and VERIFIED at start-up, attached only
 *          when the mode controller says so, and gated per-pid by deep_pids so
 *          promoting another process costs one map update, not a new attach.
 *   LSM    enforcement for Component 4. Denies execve for a process tree; the
 *          tree's existing processes keep running.
 *
 * Everything lives in one object because the tiers share maps, and sharing maps
 * between separately-loaded objects would require pinning them to bpffs. The
 * split that matters is attachment, not loading.
 *
 * Inheritance happens IN THE KERNEL, at fork. A shell or sshd under
 * investigation runs every command in a forked child, and that child execs
 * before user space has heard of it. If the gate and the denial were only
 * extended by user space, the child's exec would be neither captured nor
 * denied -- measured on the live kernel before this was fixed: 0 deep payloads
 * captured, and a denied pid could fork+exec freely.
 *
 * Build: see Makefile (needs vmlinux.h from bpftool).
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_endian.h>
#include "laax.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

/* ---------------------------------------------------------------- maps -- */

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 22);              /* 4 MiB */
} events SEC(".maps");

/* Pids (tgids) currently in deep mode. Written by user space on promotion and
   by laax_fork when a deep process forks. Entries die with the process. */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, __u32);
    __type(value, __u8);
} deep_pids SEC(".maps");

/* Pids whose execve the LSM must deny. Written by Component 4 via user space,
   and extended to children by laax_fork. Entries die with the process, so a
   recycled pid never inherits someone else's denial. */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);
    __type(value, __u8);
} denied_pids SEC(".maps");

/* Ring-buffer reservation failures. This is the event-loss metric; the
   proposal budgets under 1%, so it has to be counted, not assumed. */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} dropped SEC(".maps");

/* [0] = every bprm_check invocation, [1] = invocations that matched a denial,
   [2] = in-kernel inheritance that failed because a map was full.
   Diagnostic: distinguishes "hook never fires" from "hook fires but no match". */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 3);
    __type(key, __u32);
    __type(value, __u64);
} lsm_stats SEC(".maps");

static __always_inline void lsm_count(__u32 slot)
{
    __u64 *v = bpf_map_lookup_elem(&lsm_stats, &slot);
    if (v)
        __sync_fetch_and_add(v, 1);
}

static __always_inline void note_drop(void)
{
    __u32 k = 0;
    __u64 *v = bpf_map_lookup_elem(&dropped, &k);
    if (v)
        __sync_fetch_and_add(v, 1);
}

static __always_inline int in_deep(__u32 pid)
{
    return bpf_map_lookup_elem(&deep_pids, &pid) != NULL;
}

/* Ring-buffer memory is not zeroed, so every field is set here and each
   program only fills in what it knows. */
static __always_inline struct laax_event *reserve(__u8 kind, __u8 depth)
{
    struct laax_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) {
        note_drop();
        return NULL;
    }
    e->ts_ns    = bpf_ktime_get_ns();
    e->pid      = bpf_get_current_pid_tgid() >> 32;
    e->ppid     = 0;
    e->kind     = kind;
    e->depth    = depth;
    e->family   = 0;
    e->proto    = 0;
    e->dport    = 0;
    e->sport    = 0;
    e->argv_len = 0;
    e->_pad     = 0;
    e->daddr    = 0;
    e->saddr    = 0;
    __builtin_memset(e->daddr6, 0, sizeof(e->daddr6));
    __builtin_memset(e->saddr6, 0, sizeof(e->saddr6));
    e->flags    = 0;
    e->ret      = 0;
    e->argv[0]     = '\0';
    e->pcomm[0]    = '\0';
    e->gpcomm[0]   = '\0';
    e->filename[0] = '\0';
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    return e;
}

/* IPPROTO of the socket behind fd in the current task: connect() is the same
   syscall for TCP and UDP, and labelling a UDP "connection" as TCP is false
   telemetry. Walks task->files->fdt->fd[fd]->private_data (struct socket). */
static __always_inline __u8 fd_proto(long fd)
{
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    unsigned int max = BPF_CORE_READ(task, files, fdt, max_fds);
    if (fd < 0 || fd >= max)
        return 0;
    struct file **fds = BPF_CORE_READ(task, files, fdt, fd);
    struct file *f = NULL;
    if (bpf_probe_read_kernel(&f, sizeof(f), &fds[fd]) || !f)
        return 0;
    struct socket *sock = BPF_CORE_READ(f, private_data);
    if (!sock)
        return 0;
    short type = BPF_CORE_READ(sock, type);
    return type == 1 /* SOCK_STREAM */ ? 6 : type == 2 /* SOCK_DGRAM */ ? 17 : 0;
}

/* Addresses of a kernel socket, either family, into the event. */
static __always_inline void sock_addrs(struct laax_event *e, struct sock *sk)
{
    e->family = BPF_CORE_READ(sk, __sk_common.skc_family);
    if (e->family == LAAX_AF_INET6) {
        BPF_CORE_READ_INTO(&e->daddr6, sk, __sk_common.skc_v6_daddr.in6_u.u6_addr8);
        BPF_CORE_READ_INTO(&e->saddr6, sk, __sk_common.skc_v6_rcv_saddr.in6_u.u6_addr8);
    } else {
        e->daddr = BPF_CORE_READ(sk, __sk_common.skc_daddr);
        e->saddr = BPF_CORE_READ(sk, __sk_common.skc_rcv_saddr);
    }
}

/*
 * Submit without waking the reader on every event. A wake-up is an irq_work
 * on the submitting CPU; inside a VM that is a costly exit, and it made a
 * 300 ns program cost ~10 us whenever the reader happened to be idle
 * (measured with kernel.bpf_stats_enabled). The reader is woken only once a
 * meaningful amount of data is pending -- well before the 4 MiB buffer could
 * fill -- and otherwise drains on its own short poll timer (loader.c), which
 * bounds the added delivery latency.
 */
#define WAKEUP_BYTES (256 * 1024)

static __always_inline void submit(struct laax_event *e)
{
    long pending = bpf_ringbuf_query(&events, BPF_RB_AVAIL_DATA);
    bpf_ringbuf_submit(e, pending >= WAKEUP_BYTES ? BPF_RB_FORCE_WAKEUP : BPF_RB_NO_WAKEUP);
}

static __always_inline void fill_parent(struct laax_event *e)
{
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct task_struct *parent = BPF_CORE_READ(task, real_parent);
    if (!parent)
        return;
    e->ppid = BPF_CORE_READ(parent, tgid);
    BPF_CORE_READ_STR_INTO(&e->pcomm, parent, comm);
    /* Long-lived parents often predate the agent, so user-space process-table
       lookup cannot reliably recover this. Capture it with the exec event. */
    struct task_struct *grandparent = BPF_CORE_READ(parent, real_parent);
    if (grandparent)
        BPF_CORE_READ_STR_INTO(&e->gpcomm, grandparent, comm);
}

/* ------------------------------------------------------------- light -- */

SEC("tracepoint/sched/sched_process_exec")
int laax_exec(struct trace_event_raw_sched_process_exec *ctx)
{
    struct laax_event *e = reserve(LAAX_EXEC, LAAX_LIGHT);
    if (!e)
        return 0;
    fill_parent(e);                             /* light mode carries no argv */
    /* The path the kernel loaded, from the tracepoint's own record: a bounded
       kernel-memory copy, no user memory touched. */
    unsigned int off = ctx->__data_loc_filename & 0xFFFF;
    bpf_probe_read_kernel_str(&e->filename, sizeof(e->filename), (void *)ctx + off);
    submit(e);
    return 0;
}

/*
 * Outbound connections, IPv4 and IPv6. Reads the sockaddr from user memory at
 * syscall entry, which a racing thread could rewrite after this read (the
 * classic syscall-entry TOCTOU). Light mode accepts that for cost; deep mode's
 * tcp_sendmsg reads the kernel's own socket and is not exposed to it.
 */
SEC("tracepoint/syscalls/sys_enter_connect")
int laax_connect(struct trace_event_raw_sys_enter *ctx)
{
    void *uaddr = (void *)(long)ctx->args[1];
    __u16 family = 0;

    if (bpf_probe_read_user(&family, sizeof(family), uaddr) < 0)
        return 0;
    if (family != LAAX_AF_INET && family != LAAX_AF_INET6)
        return 0;

    struct laax_event *e = reserve(LAAX_CONNECT, LAAX_LIGHT);
    if (!e)
        return 0;
    e->family = family;
    if (family == LAAX_AF_INET) {
        struct sockaddr_in sa = {};
        if (bpf_probe_read_user(&sa, sizeof(sa), uaddr) < 0) {
            bpf_ringbuf_discard(e, 0);
            return 0;
        }
        e->daddr = sa.sin_addr.s_addr;
        e->dport = bpf_ntohs(sa.sin_port);
    } else {
        struct sockaddr_in6 sa6 = {};
        if (bpf_probe_read_user(&sa6, sizeof(sa6), uaddr) < 0) {
            bpf_ringbuf_discard(e, 0);
            return 0;
        }
        __builtin_memcpy(e->daddr6, &sa6.sin6_addr, sizeof(e->daddr6));
        e->dport = bpf_ntohs(sa6.sin6_port);
    }
    e->proto = fd_proto((long)ctx->args[0]);
    fill_parent(e);
    submit(e);
    return 0;
}

/*
 * Fork: extend the deep gate and the execve denial to the child BEFORE it can
 * run. sched_process_fork fires in kernel_clone() before wake_up_new_task(),
 * so the child cannot exec ahead of this. BTF tracepoint for the task_struct
 * pointers: a thread (pid != tgid) shares its group's tgid key already.
 *
 * Cost in light mode: two hash lookups per fork, and no event unless the
 * parent was deep or denied.
 */
SEC("tp_btf/sched_process_fork")
int BPF_PROG(laax_fork, struct task_struct *parent, struct task_struct *child)
{
    __u32 ctgid = child->tgid;
    if (child->pid != ctgid)
        return 0;                               /* new thread, not a process */

    __u32 ptgid = parent->tgid;
    __u32 flags = 0;
    __u8 one = 1;

    if (bpf_map_lookup_elem(&deep_pids, &ptgid)) {
        if (bpf_map_update_elem(&deep_pids, &ctgid, &one, BPF_ANY) == 0)
            flags |= LAAX_F_INHERIT_DEEP;
        else
            lsm_count(2);
    }
    if (bpf_map_lookup_elem(&denied_pids, &ptgid)) {
        if (bpf_map_update_elem(&denied_pids, &ctgid, &one, BPF_ANY) == 0)
            flags |= LAAX_F_INHERIT_DENY;
        else
            lsm_count(2);
    }
    if (!flags)
        return 0;

    struct laax_event *e = reserve(LAAX_FORK,
                                   (flags & LAAX_F_INHERIT_DEEP) ? LAAX_DEEP : LAAX_LIGHT);
    if (!e)
        return 0;
    e->pid   = ctgid;
    e->ppid  = ptgid;
    e->flags = flags;
    BPF_CORE_READ_STR_INTO(&e->pcomm, parent, comm);
    submit(e);
    return 0;
}

/*
 * Exit: when the LAST thread of a process exits, drop its map entries so a
 * recycled pid starts clean, and tell user space so it can forget the process.
 * Keyed on the group dying (signal->live == 0), not on the leader thread
 * exiting: otherwise pthread_exit() on the main thread would lift the gate and
 * the denial while the other threads carried on.
 */
SEC("tracepoint/sched/sched_process_exit")
int laax_exit(void *ctx)
{
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    if (BPF_CORE_READ(task, signal, live.counter) != 0)
        return 0;

    __u32 tgid = bpf_get_current_pid_tgid() >> 32;
    bpf_map_delete_elem(&deep_pids, &tgid);
    bpf_map_delete_elem(&denied_pids, &tgid);

    struct laax_event *e = reserve(LAAX_EXIT, LAAX_LIGHT);
    if (!e)
        return 0;
    e->ret = BPF_CORE_READ(task, exit_code);
    submit(e);
    return 0;
}

/* -------------------------------------------------------------- deep -- */
/*
 * Verified and loaded at start-up alongside the light programs, but left
 * unattached. The verifier has already accepted them before any threat exists,
 * so a mode switch can never fail on a verifier rejection at the worst possible
 * moment -- only bpf_link creation remains.
 */

/*
 * finalize_exec runs at the end of load_elf_binary, after the new image's
 * argument block is in place (mm->arg_start..arg_end), so the real argv can be
 * read from the new process's own memory. security_bprm_check, the previous
 * hook, fires before that block exists and only ever yielded the file name.
 */
SEC("kprobe/finalize_exec")
int BPF_KPROBE(laax_deep_exec, struct linux_binprm *bprm)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (!in_deep(pid))
        return 0;                               /* gated off: a map lookup */

    struct laax_event *e = reserve(LAAX_EXEC, LAAX_DEEP);
    if (!e)
        return 0;
    fill_parent(e);
    bpf_probe_read_kernel_str(&e->filename, sizeof(e->filename),
                              BPF_CORE_READ(bprm, filename));

    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    unsigned long start = BPF_CORE_READ(task, mm, arg_start);
    unsigned long end   = BPF_CORE_READ(task, mm, arg_end);
    __u64 len = end > start ? end - start : 0;
    if (len > MAX_ARGV_LEN - 1)
        len = MAX_ARGV_LEN - 1;
    len &= (MAX_ARGV_LEN - 1);                  /* bound for the verifier */

    if (len && bpf_probe_read_user(&e->argv, len, (void *)start) == 0) {
        e->argv[len] = '\0';
        e->argv_len = len;
    } else {
        /* Argument pages unreadable: fall back to the binary's path. */
        const char *filename = BPF_CORE_READ(bprm, filename);
        long n = bpf_probe_read_kernel_str(&e->argv, sizeof(e->argv), filename);
        e->argv_len = n > 1 ? n - 1 : 0;
    }

    submit(e);
    return 0;
}

SEC("kprobe/tcp_sendmsg")
int BPF_KPROBE(laax_deep_send, struct sock *sk, struct msghdr *msg, size_t size)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (!in_deep(pid))
        return 0;

    struct laax_event *e = reserve(LAAX_CONNECT, LAAX_DEEP);
    if (!e)
        return 0;
    sock_addrs(e, sk);
    e->proto = 6;
    e->dport = bpf_ntohs(BPF_CORE_READ(sk, __sk_common.skc_dport));
    e->sport = BPF_CORE_READ(sk, __sk_common.skc_num);
    e->ret   = (__s32)size;
    submit(e);
    return 0;
}

/* Inbound TCP: the socket inet_csk_accept returns is the new connection. The
   remote peer goes in saddr/sport, this host's service in daddr/dport. */
SEC("kretprobe/inet_csk_accept")
int BPF_KRETPROBE(laax_deep_accept, struct sock *sk)
{
    if (!sk)
        return 0;
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (!in_deep(pid))
        return 0;

    struct laax_event *e = reserve(LAAX_ACCEPT, LAAX_DEEP);
    if (!e)
        return 0;
    /* sock_addrs() writes the peer as destination; accept reverses it. */
    sock_addrs(e, sk);
    __u32 peer4 = e->daddr;
    e->daddr = e->saddr;
    e->saddr = peer4;
    __u8 tmp[16];
    __builtin_memcpy(tmp, e->daddr6, 16);
    __builtin_memcpy(e->daddr6, e->saddr6, 16);
    __builtin_memcpy(e->saddr6, tmp, 16);
    e->proto = 6;
    e->sport = bpf_ntohs(BPF_CORE_READ(sk, __sk_common.skc_dport));
    e->dport = BPF_CORE_READ(sk, __sk_common.skc_num);
    submit(e);
    return 0;
}

/* File open. Records the file's name (last path component) and open flags.
   Full paths need bpf_d_path, which kprobes may not call. */
SEC("kprobe/security_file_open")
int BPF_KPROBE(laax_deep_file, struct file *file)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (!in_deep(pid))
        return 0;

    struct laax_event *e = reserve(LAAX_FILE, LAAX_DEEP);
    if (!e)
        return 0;
    const unsigned char *name = BPF_CORE_READ(file, f_path.dentry, d_name.name);
    long n = bpf_probe_read_kernel_str(&e->argv, sizeof(e->argv), name);
    e->argv_len = n > 1 ? n - 1 : 0;
    e->ret = BPF_CORE_READ(file, f_flags);
    submit(e);
    return 0;
}

/* --------------------------------------------------------------- lsm -- */
/*
 * Enforcement arm for Component 4. Returning -EPERM denies this one execve:
 * the process keeps running, its sockets stay open, work in flight survives.
 * laax_fork extends the denial to children, so a contained shell or daemon
 * cannot launch new programs through a fork either.
 *
 * Autoload is disabled by the loader unless --lsm is passed, because on a
 * kernel without bpf in the active LSM list this program fails to LOAD, which
 * would take the whole object down with it.
 */
SEC("lsm/bprm_check_security")
int BPF_PROG(laax_deny_exec, struct linux_binprm *bprm, int ret)
{
    lsm_count(0);
    if (ret != 0)
        return ret;                             /* respect an earlier denial */

    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (bpf_map_lookup_elem(&denied_pids, &pid)) {
        lsm_count(1);
        return -1;                              /* -EPERM */
    }
    return 0;
}
