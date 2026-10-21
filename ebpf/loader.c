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
#include <pthread.h>
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
#define N_DEEP 4
static struct bpf_link *deep_links[N_DEEP];
static int deep_attached;
static void join_detach(void);

static const char *kind_name(__u8 k)
{
    switch (k) {
    case LAAX_EXEC:    return "exec";
    case LAAX_CONNECT: return "connect";
    case LAAX_ACCEPT:  return "accept";
    case LAAX_FORK:    return "fork";
    case LAAX_EXIT:    return "exit";
    case LAAX_FILE:    return "file";
    default:           return "other";
    }
}

/* Escape the first len bytes of in for JSON. argv arrives NUL-separated, so a
   NUL inside the used length becomes the separator 0x1f and user space splits
   on it; a space would be ambiguous with arguments that contain spaces. */
static void json_escape_n(const char *in, size_t len, char *out, size_t n)
{
    size_t o = 0;
    for (size_t i = 0; i < len && o + 7 < n; i++) {
        unsigned char c = (unsigned char)in[i];
        if (c == '\0') { o += snprintf(out + o, n - o, "\\u001f"); }
        else if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = c; }
        else if (c >= 0x20 && c < 0x7f) { out[o++] = c; }
    }
    out[o] = '\0';
}

static void json_escape(const char *in, char *out, size_t n)
{
    json_escape_n(in, strnlen(in, TASK_COMM_LEN), out, n);
}

static int handle_event(void *ctx, void *data, size_t len)
{
    (void)ctx;
    if (len < sizeof(struct laax_event))
        return 0;
    const struct laax_event *e = data;

    char comm[64], pcomm[64], gpcomm[64], argv[MAX_ARGV_LEN * 6 + 8];
    json_escape(e->comm, comm, sizeof(comm));
    json_escape(e->pcomm, pcomm, sizeof(pcomm));
    json_escape(e->gpcomm, gpcomm, sizeof(gpcomm));
    size_t alen = e->argv_len < MAX_ARGV_LEN ? e->argv_len : MAX_ARGV_LEN - 1;
    /* Trailing NUL of the last argument is not a separator. */
    while (alen && e->argv[alen - 1] == '\0')
        alen--;
    json_escape_n(e->argv, alen, argv, sizeof(argv));

    char filename[MAX_PATH_LEN * 6 + 8];
    json_escape_n(e->filename, strnlen(e->filename, MAX_PATH_LEN), filename, sizeof(filename));

    static const __u8 zero6[16];
    char daddr[INET6_ADDRSTRLEN] = "";
    char saddr[INET6_ADDRSTRLEN] = "";
    if (e->family == LAAX_AF_INET6) {
        if (memcmp(e->daddr6, zero6, 16)) inet_ntop(AF_INET6, e->daddr6, daddr, sizeof(daddr));
        if (memcmp(e->saddr6, zero6, 16)) inet_ntop(AF_INET6, e->saddr6, saddr, sizeof(saddr));
    } else {
        if (e->daddr) inet_ntop(AF_INET, &e->daddr, daddr, sizeof(daddr));
        if (e->saddr) inet_ntop(AF_INET, &e->saddr, saddr, sizeof(saddr));
    }
    const char *proto = e->proto == 6 ? "tcp" : e->proto == 17 ? "udp" : "";

    printf("{\"ts_ns\":%llu,\"pid\":%u,\"ppid\":%u,\"comm\":\"%s\","
           "\"pcomm\":\"%s\",\"gpcomm\":\"%s\",\"kind\":\"%s\",\"depth\":\"%s\","
           "\"family\":%u,\"proto\":\"%s\","
           "\"daddr\":\"%s\",\"saddr\":\"%s\",\"dport\":%u,\"sport\":%u,"
           "\"filename\":\"%s\",\"argv\":\"%s\",\"flags\":%u,\"ret\":%d}\n",
           (unsigned long long)e->ts_ns, e->pid, e->ppid, comm, pcomm, gpcomm,
           kind_name(e->kind), e->depth == LAAX_DEEP ? "deep" : "light",
           e->family, proto, daddr, saddr, e->dport, e->sport, filename, argv,
           e->flags, e->ret);
    fflush(stdout);
    return 0;
}

static void attach_deep(struct laax_bpf *skel)
{
    if (deep_attached) { printf("#LAT attach 0.000\n"); fflush(stdout); return; }
    join_detach();                  /* links array is the worker's until then */
    double t0 = now_ms();
    /* Programs are already loaded and verified. Only link creation happens
       here -- this is the entire cost of a mode switch. */
    struct bpf_program *progs[N_DEEP] = {
        skel->progs.laax_deep_exec, skel->progs.laax_deep_send,
        skel->progs.laax_deep_accept, skel->progs.laax_deep_file,
    };
    int ok = 1;
    for (int i = 0; i < N_DEEP; i++) {
        deep_links[i] = bpf_program__attach(progs[i]);
        if (!deep_links[i]) { ok = 0; break; }
    }
    double dt = now_ms() - t0;

    if (!ok) {
        fprintf(stderr, "attach_deep failed: %s\n", strerror(errno));
        /* All or nothing: a half-attached deep tier would report deep mode
           while silently missing whole event classes. */
        for (int i = 0; i < N_DEEP; i++) {
            bpf_link__destroy(deep_links[i]);
            deep_links[i] = NULL;
        }
        printf("#ERR attach\n");
    } else {
        deep_attached = 1;
        printf("#LAT attach %.4f\n", dt);
    }
    fflush(stdout);
}

/*
 * Detach runs on a worker thread. Each kprobe unregister waits out an RCU grace
 * period and the kernel serialises them (measured: ~95 ms per probe, 4 probes
 * ~390 ms; destroying them from parallel threads did not overlap them). Done on
 * the main thread, that would stop ring-buffer consumption for the whole time,
 * exactly during a mode transition -- where the event-loss budget applies.
 *
 * Collection does not depend on the teardown: user space removes the last pid
 * from deep_pids before asking for DETACH_DEEP, so the deep programs are
 * already no-ops. Detach only reclaims their per-call cost.
 */
static pthread_t detach_tid;
static int detach_pending;

static void destroy_deep_links(void)
{
    double t0 = now_ms();
    for (int i = 0; i < N_DEEP; i++) {
        bpf_link__destroy(deep_links[i]);
        deep_links[i] = NULL;
    }
    printf("#LAT detach %.4f\n", now_ms() - t0);
    fflush(stdout);
}

static void *detach_worker(void *_unused)
{
    (void)_unused;
    destroy_deep_links();
    return NULL;
}

static void join_detach(void)
{
    if (detach_pending) {
        pthread_join(detach_tid, NULL);
        detach_pending = 0;
    }
}

static void detach_deep(int async)
{
    if (!deep_attached) { printf("#LAT detach 0.000\n"); fflush(stdout); return; }
    deep_attached = 0;
    if (async && pthread_create(&detach_tid, NULL, detach_worker, NULL) == 0) {
        detach_pending = 1;
        return;
    }
    destroy_deep_links();
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

    __u32 k0 = 0, k1 = 1, k2 = 2;
    __u64 fired = 0, matched = 0, inherit_full = 0;
    bpf_map__lookup_elem(skel->maps.lsm_stats, &k0, sizeof(k0),
                         &fired, sizeof(fired), 0);
    bpf_map__lookup_elem(skel->maps.lsm_stats, &k1, sizeof(k1),
                         &matched, sizeof(matched), 0);
    bpf_map__lookup_elem(skel->maps.lsm_stats, &k2, sizeof(k2),
                         &inherit_full, sizeof(inherit_full), 0);
    printf("#LSM fired=%llu matched=%llu inherit_map_full=%llu\n",
           (unsigned long long)fired, (unsigned long long)matched,
           (unsigned long long)inherit_full);
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

    /* Light mode only: exec, connect, and the fork/exit pair that carries
       deep and deny state down the process tree and clears it on exit. */
    if (!bpf_program__attach(skel->progs.laax_exec) ||
        !bpf_program__attach(skel->progs.laax_connect) ||
        !bpf_program__attach(skel->progs.laax_fork) ||
        !bpf_program__attach(skel->progs.laax_exit)) {
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
        /* 10 ms: the BPF side wakes us only for large backlogs (submit() in
           laax.bpf.c), so this timer is what delivers ordinary events. It
           bounds the added latency at 10 ms against the 500 ms switch budget. */
        int n = poll(pfds, 2, 10);
        if (n < 0 && errno != EINTR) break;

        /* Unconditionally: events submitted without a wake-up never raise
           POLLIN, so waiting for it would leave them in the buffer. Consuming
           an empty ring buffer is two loads and a compare. */
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
            else if (!strcmp(line, "DETACH_DEEP"))       detach_deep(1);
            else if (sscanf(line, "GATE_ADD %u", &pid) == 1) map_op(gate_fd, pid, 1, "gate_add");
            else if (sscanf(line, "GATE_DEL %u", &pid) == 1) map_op(gate_fd, pid, 0, "gate_del");
            else if (sscanf(line, "DENY_ADD %u", &pid) == 1) map_op(deny_fd, pid, 1, "deny_add");
            else if (sscanf(line, "DENY_DEL %u", &pid) == 1) map_op(deny_fd, pid, 0, "deny_del");
            else if (!strcmp(line, "STATS"))             print_drops(skel);
            else if (!strcmp(line, "QUIT"))              break;
        }
    }

    join_detach();
    detach_deep(0);
    ring_buffer__free(rb);
    laax_bpf__destroy(skel);
    return 0;
}
