"""Project acceptance policy, stricter than TabArena's hardware portability rules."""

MIN_CPUS = 16
# The selected nominal 32 GB droplet exposes slightly less memory to the OS.
MIN_MEMORY_BYTES = 28 * 1024**3


def validate_acceptance_resources(spec, runtime):
    hardware = runtime.get("hardware", {})
    available = hardware.get("effective_cpus", 0)
    budget = spec.get("num_cpus") or available
    caps = runtime.get("thread_environment", {})
    if (
        spec.get("execution") != "upstream"
        or not runtime.get("platform", "").startswith("Linux")
        or runtime.get("machine") not in {"x86_64", "AMD64"}
        or available < MIN_CPUS
        or budget < MIN_CPUS
        or budget > available
        or hardware.get("memory_total_bytes", 0) < MIN_MEMORY_BYTES
        or any(
            value is not None and (not str(value).isdigit() or int(value) < MIN_CPUS)
            for value in caps.values()
        )
    ):
        raise ValueError(
            "acceptance resources require upstream Ray execution on Linux x86_64, "
            "at least 16 available CPUs, a CPU budget >=16 within that allocation, "
            "28 GiB usable RAM (nominal 32 GB), and no smaller numerical thread caps; "
            "reduced runs remain development diagnostics"
        )
