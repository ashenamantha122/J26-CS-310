/*
 * LAAX Component 1 -- user-space loader.
 *
 * Owns the kernel objects and nothing else. The scoring, the thresholds and the
 * mode decisions stay in Python; this program executes those decisions and
 * reports how long each one actually took.
 *
 * Protocol
 *   stdout   one JSON object per line for each event
 *            "#LAT <op> <ms>"   real timing for the last control operation
 *            "#DROP <n>"        ring-buffer reservation failures so far
 *            "#READY"           light mode attached, deep loaded but not attached
 *   stdin    ATTACH_DEEP | DETACH_DEEP
 *            GATE_ADD <pid> | GATE_DEL <pid>
 *            DENY_ADD <pid> | DENY_DEL <pid>
 *            STATS | QUIT
 *
 * The measurement that matters is ATTACH_DEEP: bpf_link creation against
 * programs the verifier accepted at start-up. That is the number the proposal
 * budgets at under 500 ms.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <poll.h>
#include <signal.h>
#include <arpa/inet.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#include "laax.h"
#include "laax.skel.h"

static volatile sig_atomic_t running = 1;
static void on_sigint(int _s) { (void)_s; running = 0; }

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Deep-mode links, created on ATTACH_DEEP and destroyed on DETACH_DEEP. */
static struct bpf_link *deep_links[2];
static int deep_attached;

static const char *kind_name(__u8 k)
{
    switch (k) {
    case LAAX_EXEC:    return "exec";
    case LAAX_CONNECT: return "connect";
    case LAAX_ACCEPT:  return "accept";
    default:           return "other";
    }
}

static void json_escape(const char *in, char *out, size_t n)
{
    size_t o = 0;
    for (size_t i = 0; in[i] && o + 2 < n; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = c; }
        else if (c >= 0x20 && c < 0x7f) { out[o++] = c; }
    }
    out[o] = '\0';
}

static int handle_event(void *ctx, void *data, size_t len)
{
    (void)ctx;
    if (len < sizeof(struct laax_event))
        return 0;
    const struct laax_event *e = data;

    char comm[64], pcomm[64], argv[512];
    json_escape(e->comm, comm, sizeof(comm));
    json_escape(e->pcomm, pcomm, sizeof(pcomm));
    json_escape(e->argv, argv, sizeof(argv));

    char daddr[INET_ADDRSTRLEN] = "";
    char saddr[INET_ADDRSTRLEN] = "";
    if (e->daddr) inet_ntop(AF_INET, &e->daddr, daddr, sizeof(daddr));
    if (e->saddr) inet_ntop(AF_INET, &e->saddr, saddr, sizeof(saddr));

    printf("{\"ts_ns\":%llu,\"pid\":%u,\"ppid\":%u,\"comm\":\"%s\","
           "\"pcomm\":\"%s\",\"kind\":\"%s\",\"depth\":\"%s\","
           "\"daddr\":\"%s\",\"saddr\":\"%s\",\"dport\":%u,"
           "\"argv\":\"%s\",\"ret\":%d}\n",
           (unsigned long long)e->ts_ns, e->pid, e->ppid, comm, pcomm,
           kind_name(e->kind), e->depth == LAAX_DEEP ? "deep" : "light",
           daddr, saddr, e->dport, argv, e->ret);
    fflush(stdout);
    return 0;
}

static void attach_deep(struct laax_bpf *skel)
{
    if (deep_attached) { printf("#LAT attach 0.000\n"); fflush(stdout); return; }
    double t0 = now_ms();
    /* Programs are already loaded and verified. Only link creation happens
       here -- this is the entire cost of a mode switch. */
    deep_links[0] = bpf_program__attach(skel->progs.laax_deep_exec);
    deep_links[1] = bpf_program__attach(skel->progs.laax_deep_send);
    double dt = now_ms() - t0;

    if (!deep_links[0] || !deep_links[1]) {
        fprintf(stderr, "attach_deep failed: %s\n", strerror(errno));
        printf("#ERR attach\n");
    } else {
        deep_attached = 1;
        printf("#LAT attach %.4f\n", dt);
    }
    fflush(stdout);
}

static void detach_deep(void)
{
    if (!deep_attached) { printf("#LAT detach 0.000\n"); fflush(stdout); return; }
    double t0 = now_ms();
    for (int i = 0; i < 2; i++) {
        bpf_link__destroy(deep_links[i]);
        deep_links[i] = NULL;
    }
    double dt = now_ms() - t0;
    deep_attached = 0;
    printf("#LAT detach %.4f\n", dt);
    fflush(stdout);
}

static void map_op(int fd, __u32 pid, int add, const char *label)
{
    __u8 one = 1;
    double t0 = now_ms();
    int err = add ? bpf_map_update_elem(fd, &pid, &one, BPF_ANY)
                  : bpf_map_delete_elem(fd, &pid);
    double dt = now_ms() - t0;
    if (err && !(!add && errno == ENOENT))
        fprintf(stderr, "%s(%u) failed: %s\n", label, pid, strerror(errno));
    printf("#LAT %s %.4f\n", label, dt);
    fflush(stdout);
}

static void print_drops(struct laax_bpf *skel)
{
    __u32 k = 0; __u64 v = 0;
    bpf_map__lookup_elem(skel->maps.dropped, &k, sizeof(k), &v, sizeof(v), 0);
    printf("#DROP %llu\n", (unsigned long long)v);

    __u32 k0 = 0, k1 = 1;
    __u64 fired = 0, matched = 0;
    bpf_map__lookup_elem(skel->maps.lsm_stats, &k0, sizeof(k0),
                         &fired, sizeof(fired), 0);
    bpf_map__lookup_elem(skel->maps.lsm_stats, &k1, sizeof(k1),
                         &matched, sizeof(matched), 0);
    printf("#LSM fired=%llu matched=%llu\n",
           (unsigned long long)fired, (unsigned long long)matched);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    int want_lsm = 0;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--lsm")) want_lsm = 1;

    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    libbpf_set_print(NULL);

    struct laax_bpf *skel = laax_bpf__open();
    if (!skel) { fprintf(stderr, "open failed\n"); return 1; }

    /* On a kernel without bpf in the active LSM list this program fails to
       LOAD and would take the whole object with it. Off unless asked for. */
    bpf_program__set_autoload(skel->progs.laax_deny_exec, want_lsm);

    /* Deep programs are loaded and VERIFIED now, attached later. */
    if (laax_bpf__load(skel)) {
        fprintf(stderr, "load failed: %s\n", strerror(errno));
        if (want_lsm)
            fprintf(stderr, "hint: --lsm needs bpf in /sys/kernel/security/lsm\n");
        laax_bpf__destroy(skel);
        return 1;
    }

    /* Light mode only. bpf_program__attach on the two tracepoints. */
    if (!bpf_program__attach(skel->progs.laax_exec) ||
        !bpf_program__attach(skel->progs.laax_connect)) {
        fprintf(stderr, "light attach failed: %s\n", strerror(errno));
        laax_bpf__destroy(skel);
        return 1;
    }
    if (want_lsm && !bpf_program__attach(skel->progs.laax_deny_exec))
        fprintf(stderr, "lsm attach failed: %s\n", strerror(errno));

    struct ring_buffer *rb =
        ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);
    if (!rb) { fprintf(stderr, "ringbuf failed\n"); laax_bpf__destroy(skel); return 1; }

    int gate_fd = bpf_map__fd(skel->maps.deep_pids);
    int deny_fd = bpf_map__fd(skel->maps.denied_pids);

    printf("#READY lsm=%d\n", want_lsm);
    fflush(stdout);

    struct pollfd pfds[2] = {
        { .fd = ring_buffer__epoll_fd(rb), .events = POLLIN },
        { .fd = STDIN_FILENO,              .events = POLLIN },
    };
    char line[256];

    while (running) {
        int n = poll(pfds, 2, 200);
        if (n < 0 && errno != EINTR) break;

        if (pfds[0].revents & POLLIN)
            ring_buffer__consume(rb);

        if (pfds[1].fd >= 0 && (pfds[1].revents & (POLLIN | POLLHUP))) {
            if (!fgets(line, sizeof(line), stdin)) {
                /* Control channel closed. Stop watching it, but KEEP COLLECTING.
                 * Exiting here would mean the agent dies whenever it is started
                 * without an attached controller -- as a daemon, from a script,
                 * with stdin at /dev/null -- and silently stops collecting
                 * telemetry. Light mode must outlive the controller. */
                pfds[1].fd = -1;
                continue;
            }
            line[strcspn(line, "\r\n")] = '\0';
            unsigned pid = 0;

            if (!strcmp(line, "ATTACH_DEEP"))            attach_deep(skel);
            else if (!strcmp(line, "DETACH_DEEP"))       detach_deep();
            else if (sscanf(line, "GATE_ADD %u", &pid) == 1) map_op(gate_fd, pid, 1, "gate_add");
            else if (sscanf(line, "GATE_DEL %u", &pid) == 1) map_op(gate_fd, pid, 0, "gate_del");
            else if (sscanf(line, "DENY_ADD %u", &pid) == 1) map_op(deny_fd, pid, 1, "deny_add");
            else if (sscanf(line, "DENY_DEL %u", &pid) == 1) map_op(deny_fd, pid, 0, "deny_del");
            else if (!strcmp(line, "STATS"))             print_drops(skel);
            else if (!strcmp(line, "QUIT"))              break;
        }
    }

    detach_deep();
    ring_buffer__free(rb);
    laax_bpf__destroy(skel);
    return 0;
}
