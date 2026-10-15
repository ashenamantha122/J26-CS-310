# Integration contract, and three problems found building it

## 1. C1 and C3 do not currently speak the same language

**The problem.** Component 1 produces eBPF syscall events. Component 3's
proposal says it trains on *"TON_IoT Linux host telemetry"*. TON_IoT's Linux
data is periodically sampled system metrics — process, memory and disk counters
read at intervals — not a syscall event stream. A C3 model trained on that data
cannot consume C1's live output, because the two have no fields in common.

Left alone, this surfaces in Phase 3 of the Gantt chart, at week 25+, when both
components are already built. That is the single largest schedule risk in the
project.

**The fix, implemented here.** `laax.contracts.HostEvent` is a canonical
host-domain record that *both* producers can emit:

```
eBPF agent  --(native)------>  HostEvent  <--(adapter)--  TON_IoT Linux CSV
```

C3 trains on adapted TON_IoT and runs live on C1 output, because both arrive as
`HostEvent`. Fields only one producer can fill (`argv`, `syscall` from eBPF;
`cpu_pct`, `rss_kb` from TON_IoT) are `Optional`, and every consumer tolerates
`None`.

**Still to write:** `services/ton_iot_adapter.py`, mapping TON_IoT Linux CSV
rows to `HostEvent(kind="sample", ...)`. Whoever writes it owns the decision
about how a sampled metric row becomes an event, and that decision constrains
what C3 can learn. It should be made deliberately, early, and by C1 and C3
together.

## 2. The transport is specified two different ways

Slide 7 (C1) and the Gantt row both say **Protocol Buffers over ZeroMQ**.
Slide 23 (C4) says **REST or gRPC with JSON policy objects**. Those are
different stacks and something has to give.

This prototype sidesteps it: components only touch the `Bus` interface, and the
backend is swappable (`InProcBus` today, `TcpBus` included, ZeroMQ later). But
the deck needs one answer before the progress presentation. Recommendation:
protobuf/ZeroMQ for the high-rate telemetry path (C1→C3, C2→C3) where message
volume matters, and REST/JSON for the low-rate control path (C3→C4, C4→C1)
where human-readable policy records and an analyst override interface matter
more. That is defensible and satisfies both slides — but say it out loud.

## 3. A hub node destroys path semantics

The first working version of C3 reported 66 correlated incidents for a single
attack. Cause: the graph joins a `host` node to every process it runs, and a
naive undirected shortest path hops
`attacker → socket → some process → host → every other process`. Every process
on the box looked compromised.

Ownership edges are not communication edges. `_incidents()` now traverses only
causal routes — attacker reaches an entry process through a socket, then the
incident propagates along `spawn` edges — and descendants are time-bounded to
those spawned after C2 flagged the device, so an `sshd` with a month of
legitimate sessions beneath it does not drag them all in.

Worth stating in the thesis: it is a property of provenance graphs generally,
not a bug specific to this build.

---

## Two design decisions in C1 worth defending in the viva

**Deep mode is inherited down the process tree.** A pivot chain is a subtree,
not a pid. A child spawned by a process under investigation must not be born
blind. Without this, the demo captures the `bash` but misses the `nc -e` and
the `chmod u+s` beneath it.

**Confidence decays on a clock, not only on events.** Without a timer the agent
has a one-way ratchet: a process that triggers deep mode and then goes quiet
emits no further events, so its EWMA never updates, so it never returns to
light. Duty cycle then rises monotonically to 100% and the entire overhead
argument collapses. `C1Agent.decay_loop` is what makes the mechanism
bidirectional, and duty cycle is the metric that proves it.

## The cost model that changes the headline number

A kprobe is attached **once, globally**, and gated by a BPF hash of pids
currently in deep mode. So the expensive operation — `bpf_link` creation against
an already-verified, already-loaded program — is paid when the agent goes from
zero deep processes to one, and again when it drops back to zero. Promoting
further processes is a single `bpf_map_update_elem`.

The prototype reports these separately (`kprobe_attach_events` vs
`map_update_ms_mean`) because collapsing them into one flat "attach cost per
process" overstates the cost of the adaptive design by roughly two orders of
magnitude — against your own argument.

---

# What the real kernel changed

Everything below was invisible until `ebpf/` compiled and ran on WSL2 kernel
6.18.33.2. Each one is a correction to a design the synthetic demo had
validated happily.

## Detach costs 13x attach, not half of it

Measured, n=30, medians: attach **13.7 ms**, detach **175.7 ms**, pid gating
**0.011 ms**. The cost model had detach *cheaper* than attach.

Detaching a kprobe must unregister it and wait out an RCU grace period;
attaching does not. Two consequences:

- `min_dwell_s` is not merely an anti-flap heuristic. Leaving deep mode is the
  expensive direction, so a dwell floor has a direct cost justification.
- The switch-latency headline must be the **worse** direction. Against the
  500 ms budget: attach has 28x headroom, detach has **2.6x**. Quoting the
  attach number alone overstates the result by an order of magnitude.

`stats()` now reports attach and detach separately for this reason.

## An EWMA cannot detect short-lived processes

On a live kernel, **270 of 277 observed processes emitted exactly one event and
exited**. That is the normal shape of Linux, and it is also the shape of
`whoami; uname; id; nc`.

A fresh process starts at confidence 0, so a single EWMA update yields
`(1-decay) * rarity`, which cannot reach `theta_up` however rare the event.
Worse, the fixed point of the EWMA under damped inheritance is `0.69 * rarity`
-- mathematically *below* the threshold, so even an unbounded chain of maximally
rare execs converges without ever firing. The scripted demo hid this completely,
because C3's `E(t)` forced the switch and the pids were long-lived.

Replaced with a one-sided **CUSUM over rarity excess**: each event contributes
its surprisal in excess of the benign baseline mean, clamped at zero. Benign
activity contributes ~0; a rare step contributes the difference; evidence
accumulates along the chain the way provenance evidence should. Confidence is
also seeded from the parent, so suspicion flows down the process tree.

## The learning window must be light-mode by construction

The CUSUM promptly produced 23 deep-mode entries during a benign phase that
must produce zero. Cause: rarity is defined relative to a baseline, and during
the learning window that baseline does not exist yet, so early events look rare
merely because little has been counted.

The controller now refuses to promote while `scorer.learning` is true. An
external alert from C3 is exempt -- that evidence does not depend on our own
uncalibrated baseline. This is worth stating as a property, not a fix: an
adaptive agent cannot act on a confidence score before its baseline exists.

## A dropped kernel field

The agent was overwriting the kernel-supplied parent comm with a local
`pid -> comm` lookup that usually missed, collapsing every lineage key to
`('?', child)` and destroying the rarity signal. `laax.bpf.c` reads
`real_parent->comm` directly, which also avoids a race against parent exit.

## The pid in a message is not the pid you think it is

`bpf_get_current_pid_tgid()` returns the pid in the **root** pid namespace.
Measured on this machine: one process was **pid 389 in the WSL distro's
namespace and pid 4984 to the kernel**.

The first version of the LSM test armed a denial using `os.getpid()`. The hook
fired 5 times, matched 0, and the exec sailed through -- containment that
reports success and does nothing. Only an invocation counter in the BPF program
distinguished "hook never fires" from "hook fires but never matches"; without
it this looks identical to a broken attach.

Why `deep_pids` worked anyway: those pids round-trip. The kernel reports a pid
in a `HostEvent`, the agent writes that same value back into the gating map, so
it is self-consistent by construction. `denied_pids` broke the moment a pid
arrived from somewhere else.

**The rule, now written into `HostEvent`:** a pid in a LAAX message is only
meaningful if C1 produced it. Never populate one from `os.getpid()`, from
`/proc` inside a container, or from an operator reading `ps`.

This is not a WSL quirk. Component 4's stated targets include cloud containers
and Kubernetes, where every workload is in its own pid namespace. Any
containment path that accepts a pid from outside C1's telemetry will fail
exactly this way, silently, in production.

## Light mode must outlive the controller

The loader exited when stdin reached EOF, so running it as a daemon, from a
script, or with stdin at `/dev/null` started it, collected nothing, and quit --
while reporting a clean exit. It now stops watching a closed control channel
and keeps consuming the ring buffer. Telemetry collection cannot be contingent
on a controller being attached.
