# Two-stream source review

Reviewed implementation at `8ec4981476e647d46287dbd1024eade75a667acf`; no independent agent review. Only cache scheduling, the CUDA registry query extension and the ordering fixture change. Quantization arithmetic and kernels are unchanged.

- Ownership: the cache owns the upload backend and both events. Sources are immutable model weights in the device's pinned host buffer type. Destination banks remain alive until the cache destructor has drained its upload backend. The destructor does not dereference the context's compute backend, which can already have been destroyed.
- Prior readers: CUDA graph evaluation joins worker streams back onto the compute stream. `end_graph` records the reader event on that stream, including failed/aborted graphs. An admission batch's copy stream wait captures this event generation before overwriting slots.
- Later readers: evictions invalidate the host map immediately and enqueue device-map invalidation behind previous compute work. Future graph work is enqueued after that invalidation. An incomplete replacement remains a miss; it is never published as a partially copied expert.
- Completion: `begin_graph` queries the copy event without a host wait, then enqueues publication in compute-stream order. Immutable scalar values back asynchronous map writes. `cudaEventQuery` distinguishes not-ready from actual CUDA errors.
- Reservation: LFU planning reserves the replacement slots immediately. While any device is still uploading, observations continue but no second admission plan can reuse those slots. Pending prompt bursts are deferred; abort clears deferred prompt accounting.
- Default/fallback: the feature defaults off, unsupported backend query or non-pinned sources log a fallback. HIP/MUSA/no-peer-copy builds do not register the extension. Existing single-stream ordering and LFU tests passed with the switch off.
- Proof limits: deterministic host fixtures cover deferred queues and event generations; ASan/UBSan cover the changed cache/test code with existing base libraries uninstrumented. This machine tests one CUDA device (sm_75); multi-device performance and non-CUDA backends were not exercised. Scheduling correctness does not establish a speed gain.
- Performance cost: completion is polled at graph boundaries; delayed publication can temporarily lower cache hit rate, and one pending device blocks further admissions on the other devices. Large prompt bursts also use the new stream. These are measured tradeoffs, not free overlap.

No source-level correctness blocker found in this scoped review. Runtime quality and performance gates remain independent.
