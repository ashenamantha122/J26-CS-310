# LAAX Component 1 -- kernel-side build.
#
#   make            build laax-loader
#   make run        build and run it (needs root)
#   make check      report whether this kernel can support the full agent
#
# Requires: clang, llvm, libbpf-dev, bpftool.
#   sudo apt install -y clang llvm libbpf-dev bpftool linux-tools-common

CLANG    ?= clang
BPFTOOL  ?= bpftool
ARCH     := $(shell uname -m | sed 's/x86_64/x86/; s/aarch64/arm64/')

BPF_CFLAGS := -O2 -g -target bpf -D__TARGET_ARCH_$(ARCH) -Wall
CFLAGS     := -O2 -g -Wall
LDLIBS     := -lbpf -lelf -lz

.PHONY: all clean run check
all: laax-loader

vmlinux.h:
	@echo "  GEN     $@"
	@test -r /sys/kernel/btf/vmlinux || \
		{ echo "ERROR: no /sys/kernel/btf/vmlinux -- kernel lacks CONFIG_DEBUG_INFO_BTF"; exit 1; }
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

laax.bpf.o: laax.bpf.c laax.h vmlinux.h
	@echo "  BPF     $@"
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

laax.skel.h: laax.bpf.o
	@echo "  SKEL    $@"
	$(BPFTOOL) gen skeleton $< > $@

laax-loader: loader.c laax.skel.h laax.h
	@echo "  CC      $@"
	$(CLANG) $(CFLAGS) loader.c -o $@ $(LDLIBS)

run: laax-loader
	sudo ./laax-loader

# Is this kernel capable of running the full agent, including enforcement?
check:
	@echo "kernel:      $$(uname -r)"
	@printf "BTF:         "; test -r /sys/kernel/btf/vmlinux && echo "yes" || echo "NO -- CO-RE unavailable"
	@printf "BPF_LSM:     "; (zcat /proc/config.gz 2>/dev/null || cat /boot/config-$$(uname -r) 2>/dev/null) \
		| grep -q "CONFIG_BPF_LSM=y" && echo "compiled in" || echo "NO"
	@printf "lsm active:  "; cat /sys/kernel/security/lsm 2>/dev/null || echo "securityfs not mounted"
	@printf "bpf in lsm:  "; grep -q bpf /sys/kernel/security/lsm 2>/dev/null \
		&& echo "yes -- enforcement available" \
		|| echo "NO -- add 'lsm=...,bpf' to the kernel cmdline for --lsm"
	@printf "clang:       "; $(CLANG) --version 2>/dev/null | head -1 || echo "missing"
	@printf "bpftool:     "; $(BPFTOOL) version 2>/dev/null | head -1 || echo "missing"

clean:
	rm -f laax-loader laax.bpf.o laax.skel.h vmlinux.h
