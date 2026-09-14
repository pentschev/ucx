# Host-via-CUDA Rendezvous Staging Design

## Summary

Add an opt-in UCP policy that prefers CUDA-backed rendezvous fragments for
tagged transfers involving host memory. The policy enables these staged data
paths when CUDA IPC is reachable:

- host to CUDA: host A to CUDA fragment A to CUDA B;
- CUDA to host: CUDA A to CUDA fragment B to host B;
- host to host: host A to CUDA fragment A to CUDA fragment B to host B.

The policy works both within a node and across nodes on systems where the
`cuda_ipc` transport supports Multi-Node NVLink (MNNVL). It is disabled by
default and falls back to normal rendezvous selection when the complete staged
path is unavailable.

## Goals

- Allow tag rendezvous transfers involving host memory to use NVLink through
  CUDA IPC when that path is faster than the normal host transport.
- Apply the policy symmetrically to host-to-CUDA, CUDA-to-host, and
  host-to-host transfers.
- Support conventional same-node CUDA IPC and cross-node MNNVL without
  hard-coding endpoint locality in UCP policy.
- Route all staging allocations through the existing rendezvous fragment
  pools so `RNDV_FRAG_WORKER_MAX_MEM` throttles each participating worker.
- Provide an `ucx_perftest` recipe that forces the policy and exposes the
  selected protocol.

## Non-goals

- Changing default protocol selection when the option is disabled.
- Applying the policy to Active Messages, RMA, stream, or tag eager traffic.
- Failing a transfer when CUDA IPC, MNNVL, or an exportable CUDA allocation is
  unavailable.
- Changing the semantics or default value of `RNDV_FRAG_WORKER_MAX_MEM`.
- Adding a public API or changing a rendezvous wire format.
- Adding MNNVL support to the CUDA IPC transport; the policy consumes the
  transport's existing reachability and fabric-handle support.

## Configuration

Add this UCP context option:

```text
RNDV_PIPELINE_HOST_CUDA_STAGING_FORCE=n
```

The corresponding context field is
`rndv_pipeline_host_cuda_staging_force`. The option is effective only when all
of the following hold:

- protocol v2 is enabled;
- `RNDV_SCHEME=auto`;
- `RNDV_PIPELINE_SHM_ENABLE=y` (the existing name is retained even though the
  pipeline can use MNNVL);
- `RNDV_FRAG_MEM_TYPES` contains `cuda`;
- the operation is tagged rendezvous traffic;
- one or both application buffers use host memory and the other buffer is
  either host or CUDA memory; and
- protocol probing finds every copy and CUDA IPC operation required by the
  staged path.

The option does not require `RNDV_FRAG_WORKER_MAX_MEM` to be finite. When the
user configures a finite value, every fragment used by this policy honors it.
An explicit non-auto `RNDV_SCHEME` takes precedence and disables the policy.

## Protocol Selection

The implementation extends the existing tag-rendezvous provenance mechanism
used by `RNDV_PIPELINE_SHM_CUDA_STAGING_FORCE`. A shared helper determines
whether either force policy is active so the tag-only marker is propagated
through RTS/RTR handling and pipeline-fragment selection. Separate predicates
then preserve the exact scope of each policy:

- the existing policy remains limited to intra-node CUDA-to-CUDA transfers
  staged through attached host memory;
- the new policy accepts host-to-host, host-to-CUDA, and CUDA-to-host pairs and
  does not check the endpoint's intra-node flag.

For the new policy, only rendezvous variants whose staging memory type is CUDA
are promoted. Direct protocols and variants with host fragments remain
available as fallback, but do not compete with a viable forced CUDA-staging
variant. If no complete forced variant is available, the existing last-resort
selection rules choose a normal rendezvous protocol.

Protocol estimation must use the properties of the CUDA allocation selected
for the fragment pool. `ucp_mm_get_alloc_md_index()` already probes a
representative allocation and caches its memory flags. On an inter-node
endpoint, the CUDA IPC memory domain is eligible only when those flags include
`UCS_MEM_FLAG_MEMTYPE_COPY_INTER_NODE`. Runtime fragment key packing repeats
the real registration/exportability checks. Consequently, UCP does not need a
separate MNNVL hardware predicate: CUDA IPC reachability and allocation flags
are the source of truth.

The change introduces no new protocol-select flag on the wire. Any additional
internal selection flag or rkey-configuration identity is local-only and must
not alter packed headers.

## Data Paths

### Host to CUDA

The sender selects the PUT memory-type child with CUDA fragments. It acquires
a local CUDA fragment, copies the source host slice into it through the
memory-type endpoint, and puts the fragment to the receiver's CUDA buffer over
the selected CUDA IPC lane. Completion releases the sender fragment.

### CUDA to Host

The receiver selects the RTR memory-type child with CUDA fragments. It acquires
a CUDA fragment and publishes its address and key in RTR. The sender transfers
the source CUDA slice to that fragment over CUDA IPC. The receiver then copies
the fragment into the destination host slice and releases it.

### Host to Host

The receiver uses the CUDA-fragment RTR path and the sender responds with the
CUDA-fragment PUT path. Thus both workers independently acquire a fragment.
The sender copies host to local CUDA, CUDA IPC moves the slice between the two
fragments, and the receiver copies remote CUDA to host. Each fragment is held
until the operation that consumes it completes.

## Fragment Throttling and Lifetime

No allocation bypass is added. Sender PUT staging and receiver RTR staging call
`ucp_proto_rndv_mtype_request_init()`, which obtains a descriptor from the
worker's fragment mpool keyed by memory type and system device. The mpool's
`max_elems` derives from `RNDV_FRAG_WORKER_MAX_MEM`, `RNDV_FRAG_SIZE`, and
`RNDV_FRAG_ALLOC_COUNT`.

When a finite cap is exhausted, allocation returns `UCS_ERR_NO_RESOURCE` and
the request enters the existing worker flow-control queues. PUT/GET work keeps
priority over RTR work to favor operations that release memory. Returning a
fragment schedules pending work. Normal completion, reset, cancellation, and
abort paths must each return an acquired descriptor exactly once and remove
queued requests without leaving callbacks behind.

The cap applies independently to each worker and, as in the existing
implementation, independently to each fragment memory-type/system-device
pool. A host-to-host transfer can therefore hold bounded CUDA staging memory
on both workers when the user supplies a finite cap.

## Fallback and Error Handling

The configuration is a preference among available protocols, not a semantic
requirement for CUDA IPC. These conditions make the forced variant unavailable
and allow normal rendezvous fallback:

- no CUDA device or CUDA fragment allocator;
- CUDA absent from `RNDV_FRAG_MEM_TYPES`;
- no reachable CUDA IPC lane;
- cross-node CUDA IPC without MNNVL reachability;
- a CUDA allocation that cannot be exported with a fabric handle;
- failure to register or pack the required fragment key; or
- an explicit `RNDV_SCHEME` selection.

Errors after a staged protocol starts use its existing request-abort and
fragment-release paths. The policy must not retry the same transfer with a
different protocol after partial data movement.

## Observability and Perftest Sample

No new `ucx_perftest` option is required. Its existing `-m SEND,RECV` argument
selects the application buffer types, while UCP environment variables enforce
the protocol policy. Documentation will provide commands equivalent to:

```sh
UCX_TLS=rc,cuda_copy,cuda_ipc \
UCX_PROTO_ENABLE=y \
UCX_PROTO_INFO=y \
UCX_RNDV_PIPELINE_HOST_CUDA_STAGING_FORCE=y \
UCX_RNDV_FRAG_MEM_TYPES=cuda \
UCX_RNDV_FRAG_WORKER_MAX_MEM=256M \
ucx_perftest -t tag_bw -m host,host
```

Replace `host,host` with `host,cuda` or `cuda,host` for the asymmetric cases.
For MNNVL, both peers must also have CUDA IPC MNNVL enabled and a working IMEX
fabric configuration. `UCX_PROTO_INFO=y` must show CUDA-fragment pipeline
children and a `cuda_ipc` bulk lane; this distinguishes successful enforcement
from normal fallback.

## Testing

Protocol-selection tests will cover:

- the option is disabled by default;
- forced host-to-host, host-to-CUDA, and CUDA-to-host selection;
- only CUDA fragment variants are promoted;
- the selected protocol description contains CUDA copy stages and
  `cuda_ipc` for the bulk transfer;
- explicit `RNDV_SCHEME` disables the preference;
- AM and RMA selections do not receive tag provenance;
- absence of a viable CUDA IPC child falls back to a normal protocol;
- same-node eligibility does not depend on synthetic MNNVL state; and
- an inter-node candidate is eligible only with CUDA IPC reachability and an
  inter-node-exportable CUDA fragment allocation.

A functional GPU-aware tag test will transfer and validate data for the three
memory-type pairs. At least one multi-fragment case will configure a small
finite `RNDV_FRAG_WORKER_MAX_MEM`, verify that the throttling counter advances,
and verify that both workers' fragment flow-control queues are empty after
completion. Where CI lacks two GPUs, NVLink, or MNNVL hardware, mock protocol
tests provide deterministic selection coverage and hardware tests skip with a
specific reason.

## Compatibility and Performance

The default-disabled branch must add no request-time allocation and no new
wire data. Policy checks run during protocol initialization/selection, not for
each fragment copy. No endpoint or rkey public structure is enlarged. Existing
CUDA-via-host force behavior and configuration remain unchanged.

The feature intentionally forces a potentially slower protocol when enabled;
users are expected to validate it on their topology with `ucx_perftest`. Normal
automatic performance selection remains unchanged when it is disabled.

## Acceptance Criteria

- With the option disabled, protocol selection matches current behavior.
- With the option enabled and a viable CUDA IPC path, tag rendezvous selects
  the CUDA-fragment staged path for all three supported memory-type pairs.
- The same policy can select CUDA IPC over MNNVL without an intra-node check.
- Without a viable complete path, transfers succeed through normal fallback.
- A finite `RNDV_FRAG_WORKER_MAX_MEM` bounds every staging pool used by the
  transfer and exhausted pools make progress through existing throttling.
- Functional tests validate transferred data and leave no fragment requests or
  flow-control callbacks queued.
- Documented `ucx_perftest` commands can force and identify the path.
