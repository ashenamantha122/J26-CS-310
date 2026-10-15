/*
 * LAAX Component 1 -- kernel side.
 *
 * One object, three tiers of program, attached independently:
 *
 *   LIGHT  static tracepoints. Attached at start-up, never detached. Stable
 *          ABI, cheaper than kprobes, carries no payload.
 *   DEEP   kprobes/kretprobes. Loaded and VERIFIED at start-up, attached only
 *          when the mode controller says so, and gated per-pid by deep_pids so
 *          promoting another process costs one map update, not a new attach.
 *   LSM    enforcement for Component 4. Denies one execve; the parent process
 *          survives.
 *
 * Everything lives in one object because the tiers share maps, and sharing maps
 * between separately-loaded objects would require pinning them to bpffs. The
 * split that matters is attachment, not loading.
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

/* Pids currently in deep mode. Written by user space on every promotion. */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, __u32);
    __type(value, __u8);
} deep_pids SEC(".maps");

/* Pids whose execve the LSM must deny. Written by Component 4. */
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

/* [0] = every bprm_check invocation, [1] = invocations that matched a denial.
   Diagnostic: distinguishes "hook never fires" from "hook fires but no match". */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 2);
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

static __always_inline struct laax_event *reserve(void)
{
    struct laax_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e)
        note_drop();
    return e;
}

/* ------------------------------------------------------------- light -- */

SEC("tracepoint/sched/sched_process_exec")
int laax_exec(struct trace_event_raw_sched_process_exec *ctx)
{
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct laax_event *e = reserve();
    if (!e)
        return 0;

    e->ts_ns = bpf_ktime_get_ns();
    e->pid   = bpf_get_current_pid_tgid() >> 32;
    e->ppid  = BPF_CORE_READ(task, real_parent, tgid);
    e->kind  = LAAX_EXEC;
    e->depth = LAAX_LIGHT;
    e->dport = 0;
    e->daddr = 0;
    e->saddr = 0;
    e->ret   = 0;
    e->argv[0] = '\0';                          /* light mode carries no argv */
    bpf_get_current_comm(&e->comm, sizeof(e->comm));
    BPF_CORE_READ_STR_INTO(&e->pcomm, task, real_parent, comm);

    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("tracepoint/syscalls/sys_enter_connect")
int laax_connect(struct trace_event_raw_sys_enter *ctx)
{
    struct sockaddr_in sa = {};
    void *uaddr = (void *)(long)ctx->args[1];

    if (bpf_probe_read_user(&sa, sizeof(sa), uaddr) < 0)
        return 0;
    if (sa.sin_family != 2 /* AF_INET */)
        return 0;

    struct laax_event *e = reserve();
    if (!e)
        return 0;

    e->ts_ns = bpf_ktime_get_ns();
    e->pid   = bpf_get_current_pid_tgid() >> 32;
    e->ppid  = 0;
    e->kind  = LAAX_CONNECT;
    e->depth = LAAX_LIGHT;
    e->daddr = sa.sin_addr.s_addr;
    e->saddr = 0;
    e->dport = bpf_ntohs(sa.sin_port);
    e->ret   = 0;
    e->argv[0]  = '\0';
    e->pcomm[0] = '\0';
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/* -------------------------------------------------------------- deep -- */
/*
 * Verified and loaded at start-up alongside the light programs, but left
 * unattached. The verifier has already accepted them before any threat exists,
 * so a mode switch can never fail on a verifier rejection at the worst possible
 * moment -- only bpf_link creation remains, and that is single-digit ms.
 */

SEC("kprobe/security_bprm_check")
int BPF_KPROBE(laax_deep_exec, struct linux_binprm *bprm)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (!in_deep(pid))
        return 0;                               /* gated off: a map lookup */

    struct laax_event *e = reserve();
    if (!e)
        return 0;

    e->ts_ns = bpf_ktime_get_ns();
    e->pid   = pid;
    e->ppid  = 0;
    e->kind  = LAAX_EXEC;
    e->depth = LAAX_DEEP;
    e->dport = 0;
    e->daddr = 0;
    e->saddr = 0;
    e->ret   = 0;
    e->pcomm[0] = '\0';
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    const char *filename = BPF_CORE_READ(bprm, filename);
    bpf_probe_read_kernel_str(&e->argv, sizeof(e->argv), filename);

    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("kprobe/tcp_sendmsg")
int BPF_KPROBE(laax_deep_send, struct sock *sk, struct msghdr *msg, size_t size)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    if (!in_deep(pid))
        return 0;

    struct laax_event *e = reserve();
    if (!e)
        return 0;

    e->ts_ns = bpf_ktime_get_ns();
    e->pid   = pid;
    e->ppid  = 0;
    e->kind  = LAAX_CONNECT;
    e->depth = LAAX_DEEP;
    e->daddr = BPF_CORE_READ(sk, __sk_common.skc_daddr);
    e->saddr = BPF_CORE_READ(sk, __sk_common.skc_rcv_saddr);
    e->dport = bpf_ntohs(BPF_CORE_READ(sk, __sk_common.skc_dport));
    e->ret   = (__s32)size;
    e->argv[0]  = '\0';
    e->pcomm[0] = '\0';
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    bpf_ringbuf_submit(e, 0);
    return 0;
}

/* --------------------------------------------------------------- lsm -- */
/*
 * Enforcement arm for Component 4. Returning -EPERM denies this one execve:
 * the parent keeps running, its sockets stay open, work in flight survives.
 * That is the difference between surgical containment and killing a subtree.
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
