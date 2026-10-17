// hook_skeleton.bpf.c — minimal LSM hook attach point for Component 4.
//
// This attaches to an EXISTING LSM hook (bprm_check_security, fired on
// every exec()) rather than inventing a new one — matches your proposal's
// "attaching to existing LSM hook points, not building new ones".
//
// Deny decision is driven entirely by a BPF hash map keyed by PID, which
// is what gives you dynamic, graduated targeting (your "minimum effective
// response" novelty claim) instead of a single global on/off switch:
// the Decision Engine's RESTRICT level writes exactly one PID/host into
// this map; ISOLATE can write many. Removing the key is the rollback —
// no reload, no restart, matching Component 1's own "no restart, no
// policy redeployment" design language so the two components speak the
// same architectural idiom.
//
// Build: clang -O2 -g -target bpf -c hook_skeleton.bpf.c -o hook_skeleton.bpf.o
// (requires libbpf headers + BTF — see check_ebpf_kernel.sh)

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "GPL";

// Map: PID -> deny flag (1 = deny exec). Populated/cleared by the
// user-space control script (ebpf_enforcer.py) in response to Decision
// Engine output. BPF_MAP_TYPE_HASH so lookups stay O(1) regardless of
// how many targets are under containment at once.
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u32);    // pid
    __type(value, __u8);   // 1 = deny
} denied_pids SEC(".maps");

// Ring buffer for reporting enforcement events back to user space
// (so the SOC-facing audit log shows exactly which exec was blocked,
// when, for which PID — your "explainable policy records" claim).
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

struct enforcement_event {
    __u32 pid;
    __u64 timestamp_ns;
    __u8  action;   // 1 = denied exec
};

SEC("lsm/bprm_check_security")
int BPF_PROG(restrict_exec, struct linux_binprm *bprm)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u8 *denied = bpf_map_lookup_elem(&denied_pids, &pid);

    if (denied && *denied == 1) {
        struct enforcement_event *evt = bpf_ringbuf_reserve(&events, sizeof(*evt), 0);
        if (evt) {
            evt->pid = pid;
            evt->timestamp_ns = bpf_ktime_get_ns();
            evt->action = 1;
            bpf_ringbuf_submit(evt, 0);
        }
        return -1; // deny: LSM hooks return non-zero to block the operation
    }
    return 0; // allow
}
