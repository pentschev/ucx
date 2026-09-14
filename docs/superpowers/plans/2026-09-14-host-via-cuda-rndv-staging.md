# Host-via-CUDA Rendezvous Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-disabled tag-rendezvous policy that forces CUDA-fragment staging for host-to-host and host-to/from-CUDA transfers over reachable same-node CUDA IPC or cross-node MNNVL paths.

**Architecture:** Extend the existing CUDA-via-host force framework with a second, independent host-via-CUDA policy. Preserve tag provenance through rendezvous control selection, promote only complete CUDA-fragment protocol variants, and rely on existing UCT reachability, CUDA allocation exportability, and rendezvous fragment mpools for fallback and throttling.

**Tech Stack:** UCX UCP protocol v2 in C, UCT CUDA IPC/fabric capabilities, UCX GoogleTest C++11 suite, Sphinx Markdown/reStructuredText documentation, `ucx_perftest`.

**Spec:** `docs/superpowers/specs/2026-09-14-host-via-cuda-rndv-staging-design.md`

## Global Constraints

- The option is disabled by default and affects only tag rendezvous traffic.
- Support `host->host`, `host->cuda`, and `cuda->host` symmetrically.
- Do not require the endpoint to be intra-node; CUDA IPC reachability and `UCS_MEM_FLAG_MEMTYPE_COPY_INTER_NODE` decide MNNVL eligibility.
- Fall back to normal rendezvous selection when the staged path is incomplete.
- Reuse existing fragment mpools and `RNDV_FRAG_WORKER_MAX_MEM`; do not add an allocator or accounting path.
- Do not change public API, rendezvous wire formats, or default protocol selection.
- Keep the feature patch below 500 added lines, excluding the already committed design and this implementation plan.

---

### Task 1: Add Configuration and Host-via-CUDA Policy Predicates

**Files:**
- Modify: `src/ucp/core/ucp_context.h`
- Modify: `src/ucp/core/ucp_context.c`
- Modify: `src/ucp/rndv/proto_rndv.h`
- Modify: `src/ucp/rndv/proto_rndv.inl`
- Test: `test/gtest/ucp/test_ucp_proto.cc`

**Interfaces:**
- Consumes: `ucp_proto_select_op_flags()`, `ucp_proto_init_check_op()`, and the existing CUDA-staging force helpers.
- Produces: context field `int rndv_pipeline_host_cuda_staging_force`; `ucp_proto_rndv_staging_force_enabled(ucp_context_h)`; `ucp_proto_rndv_host_cuda_staging_force(const ucp_proto_init_params_t *)`; internal control flag `UCP_PROTO_RNDV_CTRL_FLAG_FORCE_CUDA_FRAG_CHILD`.

- [ ] **Step 1: Add failing policy-scope tests**

Extend the existing `test_ucp_proto_rndv_force_cuda` fixture with a helper that constructs the local/remote memory pair through `select_rndv_send_protocol()`. Add tests equivalent to:

```cpp
UCS_TEST_P(test_ucp_proto_rndv_force_cuda,
           rndv_force_host_cuda_scope,
           "RNDV_PIPELINE_HOST_CUDA_STAGING_FORCE=y",
           "RNDV_FRAG_MEM_TYPES=cuda")
{
    for (auto pair : {{UCS_MEMORY_TYPE_HOST, UCS_MEMORY_TYPE_HOST},
                      {UCS_MEMORY_TYPE_HOST, UCS_MEMORY_TYPE_CUDA},
                      {UCS_MEMORY_TYPE_CUDA, UCS_MEMORY_TYPE_HOST}}) {
        EXPECT_TRUE(should_force_host_cuda_pair(pair.first, pair.second));
    }

    EXPECT_FALSE(should_force_host_cuda_pair(UCS_MEMORY_TYPE_CUDA,
                                             UCS_MEMORY_TYPE_CUDA));
}

UCS_TEST_P(test_ucp_proto_rndv_force_cuda,
           rndv_force_host_cuda_is_not_intra_node_only,
           "RNDV_PIPELINE_HOST_CUDA_STAGING_FORCE=y",
           "RNDV_FRAG_MEM_TYPES=cuda")
{
    EXPECT_TRUE(should_force_host_cuda_pair(UCS_MEMORY_TYPE_HOST,
                                            UCS_MEMORY_TYPE_CUDA,
                                            UCP_EP_CONFIG_KEY_FLAG_INTER_NODE));
}
```

The fixture helper must invoke the real inline policy predicate with a complete `ucp_proto_init_params_t`; it must not reproduce the boolean expression in test code. Also add negative cases for an untagged operation, proto v1, `RNDV_SCHEME=get_zcopy`, and CUDA absent from `RNDV_FRAG_MEM_TYPES`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```sh
make -C build/test/gtest -j$(nproc) gtest
build/test/gtest/gtest --gtest_filter='*test_ucp_proto_rndv_force_cuda*rndv_force_host_cuda*'
```

Expected: compilation fails because the context field and host-via-CUDA policy predicate do not exist. After adding only declarations needed to compile, the tests fail because the predicate returns false.

- [ ] **Step 3: Add the minimal configuration and predicates**

Add the context field next to the existing staging option and this config entry next to `RNDV_PIPELINE_SHM_CUDA_STAGING_FORCE`:

```c
{"RNDV_PIPELINE_HOST_CUDA_STAGING_FORCE", "n",
 "Prefer CUDA-fragment staging for host-to-host and host-to/from-CUDA tag\n"
 "rendezvous transfers when RNDV_SCHEME is auto and the complete CUDA IPC\n"
 "path is available. CUDA IPC reachability may be intra-node or MNNVL.\n"
 "Other rendezvous protocols remain as fallback.",
 ucs_offsetof(ucp_context_config_t,
              rndv_pipeline_host_cuda_staging_force),
 UCS_CONFIG_TYPE_BOOL},
```

Refactor the current master enable check without changing the existing policy:

```c
static UCS_F_ALWAYS_INLINE int
ucp_proto_rndv_staging_force_enabled(ucp_context_h context)
{
    const ucp_context_config_t *cfg = &context->config.ext;

    return cfg->proto_enable && cfg->rndv_shm_ppln_enable &&
           (cfg->rndv_mode == UCP_RNDV_MODE_AUTO) &&
           (cfg->rndv_shm_cuda_staging_force ||
            cfg->rndv_pipeline_host_cuda_staging_force);
}
```

Keep `ucp_proto_rndv_shm_pipeline_force_enabled()` as the old option-specific predicate. Add a host policy predicate that requires tag provenance, the requested RNDV operation, a non-null rkey key, CUDA in `rndv_frag_mem_types`, and exactly these unordered memory pairs:

```c
((local_type == UCS_MEMORY_TYPE_HOST) &&
 ((remote_type == UCS_MEMORY_TYPE_HOST) ||
  (remote_type == UCS_MEMORY_TYPE_CUDA))) ||
((local_type == UCS_MEMORY_TYPE_CUDA) &&
 (remote_type == UCS_MEMORY_TYPE_HOST))
```

Do not test `UCP_EP_CONFIG_KEY_FLAG_INTRA_NODE`. Add a distinct internal control flag for selecting CUDA-fragment children so the old host-fragment flag retains its behavior.

- [ ] **Step 4: Run the policy tests and verify GREEN**

Run the focused command from Step 2. Expected: all new scope cases pass, including the inter-node endpoint-key case, while CUDA-to-CUDA and untagged cases remain false.

- [ ] **Step 5: Run existing force-policy regression tests**

Run:

```sh
build/test/gtest/gtest --gtest_filter='*test_ucp_proto_rndv_force_cuda*:*test_ucp_proto_rma_rndv*:*test_ucp_proto_am_rndv*'
```

Expected: PASS, with hardware-dependent cases reported as explicit skips rather than failures.

- [ ] **Step 6: Commit Task 1**

```sh
git add src/ucp/core/ucp_context.c src/ucp/core/ucp_context.h \
        src/ucp/rndv/proto_rndv.h src/ucp/rndv/proto_rndv.inl \
        test/gtest/ucp/test_ucp_proto.cc
git commit -m "UCP/RNDV: Add host CUDA staging policy"
```

---

### Task 2: Force Complete CUDA-Fragment Rendezvous Variants

**Files:**
- Modify: `src/ucp/rndv/proto_rndv.c`
- Modify: `src/ucp/rndv/proto_rndv.inl`
- Modify: `src/ucp/rndv/rndv_ppln.c`
- Modify: `src/ucp/rndv/rndv_rtr.c`
- Modify: `src/ucp/rndv/rndv_put.c`
- Test: `test/gtest/ucp/test_ucp_proto.cc`
- Test: `test/gtest/ucp/test_ucp_proto_mock.cc`

**Interfaces:**
- Consumes: Task 1 predicates and `UCP_PROTO_RNDV_CTRL_FLAG_FORCE_CUDA_FRAG_CHILD`; existing `ucp_proto_rndv_cfg_thresh()`, control-variant threshold/priority helpers, fragment protocol selection, `ucp_mm_get_alloc_md_index()` allocation flags, and `ucp_proto_rndv_mtype_init()`.
- Produces: forced tag-RNDV protocol configurations whose queried descriptions contain CUDA fragments and `cuda_ipc`; normal fallback thresholds when no complete candidate exists; functional coverage of data integrity and fragment throttling.

- [ ] **Step 1: Add failing protocol-description tests for all directions**

In `test_ucp_proto_mock.cc`, add one table-driven test to
`test_ucp_proto_mock_cuda_ipc` using real tag protocol lookup and these
expectations:

```cpp
struct case_t {
    ucs_memory_type_t send_type;
    ucs_memory_type_t recv_type;
    const char        *required_stage;
};

const case_t cases[] = {
    {UCS_MEMORY_TYPE_HOST, UCS_MEMORY_TYPE_CUDA, "frag cuda"},
    {UCS_MEMORY_TYPE_CUDA, UCS_MEMORY_TYPE_HOST, "frag cuda"},
    {UCS_MEMORY_TYPE_HOST, UCS_MEMORY_TYPE_HOST, "frag cuda"}
};
```

For every case, select the tag-RNDV envelope with the corresponding remote rkey memory type and assert that the queried configuration contains `rendezvous pipeline`, `frag cuda`, `cuda_copy`, and `cuda_ipc`. Assert that CUDA-to-CUDA remains outside this new policy. Add a disabled-default test that verifies the protocol is not forcibly replaced.

- [ ] **Step 2: Add the failing functional throttling test**

Add a tag send/receive helper to `test_ucp_proto_mock_cuda_ipc` that accepts
independent send and receive memory types, uses `ucp_tag_recv_nbx()` and
`ucp_tag_send_nbx()`, waits for both requests, and validates a `mem_buffer`
pattern. Run it for `host:host`, `host:cuda`, and `cuda:host` with:

```cpp
modify_config("RNDV_THRESH", "0");
modify_config("RNDV_PIPELINE_HOST_CUDA_STAGING_FORCE", "y");
modify_config("RNDV_FRAG_MEM_TYPES", "cuda");
modify_config("RNDV_FRAG_SIZE", "cuda:64K");
modify_config("RNDV_FRAG_ALLOC_COUNT", "cuda:2");
modify_config("RNDV_FRAG_WORKER_MAX_MEM", "128K");
```

Add `check_pending_queues_empty(entity&)` that iterates
`UCP_WORKER_RNDV_FC_OP_LAST`. Transfer 256 KiB, validate the destination,
use direction-specific PUT/RTR counter deltas, assert that the workers'
`UCP_WORKER_STAT_RNDV_MTYPE_FC_THROTTLED` total increases, and assert that both
workers' queues are empty. Add reciprocal host-to-host transfers posted before
either direction completes: a one-fragment cap must complete through normal
fallback, while a two-fragment cap must stage, throttle, and complete.

- [ ] **Step 3: Run selection and functional tests and verify RED**

Run:

```sh
make -C build/test/gtest -j$(nproc) gtest
build/test/gtest/gtest --gtest_filter='*host_cuda*'
```

Expected: the selected descriptions use the existing direct/host path or a
stub instead of the required CUDA-fragment pipeline, and the functional test's
throttling assertion fails. A build without CUDA or usable CUDA IPC may skip
the functional case with an explicit capability reason, but the selection
assertion must fail before implementation.

- [ ] **Step 4: Propagate tag provenance whenever either force option is active**

Replace call sites that conditionally attach `UCP_PROTO_SELECT_OP_FLAG_TAG_RNDV` based only on `ucp_proto_rndv_shm_pipeline_force_enabled()` with `ucp_proto_rndv_staging_force_enabled()`. Keep opcode mapping local-only and retain existing AM/RMA zero-flag behavior. The affected paths are RTS receive selection, RTR response selection, and pipeline child-key construction.

- [ ] **Step 5: Promote only CUDA-fragment candidates**

Extend `ucp_proto_rndv_cfg_thresh()`, `ucp_proto_rndv_ctrl_init_flags()`, and the control variant threshold helper so that:

```c
if (ucp_proto_rndv_host_cuda_staging_force(init_params)) {
    return (rndv_modes & UCS_BIT(UCP_RNDV_MODE_PUT_PIPELINE)) ?
           UCS_MEMUNITS_AUTO : UCS_MEMUNITS_INF;
}
```

The CUDA-child control flag is set when a selected child uses a CUDA fragment.
A control envelope is promoted only if its remote child remains selectable and
carries the matching CUDA-child flag. Host-fragment variants and incomplete
CUDA paths retain `UCS_MEMUNITS_INF` so they can serve only as normal
last-resort fallbacks.

For host-to-CUDA, promote the receiver's plain RTR control path only when its
remote PUT child is the CUDA-fragment child. This yields the required
host A -> CUDA fragment A -> CUDA B path without allocating an RTR fragment at
CUDA B. CUDA-to-host uses only the receiver RTR fragment. Host-to-host publishes
a receiver CUDA fragment key, and the existing exact-rkey-type preference makes
the sender choose a CUDA fragment as well.

- [ ] **Step 6: Preserve inter-node exportability in protocol estimation**

When building the synthetic rkey configuration for a forced CUDA-fragment
child, propagate the representative CUDA allocation's
`UCS_MEM_FLAG_MEMTYPE_COPY_INTER_NODE` from the `ucp_memory_info_t` returned by
`ucp_mm_get_alloc_md_index()`. Do not infer this bit for arbitrary user
buffers. This makes a cross-node candidate selectable only when the fragment
allocator itself produces fabric-exportable CUDA memory; runtime rkey packing
continues to validate the real fragment memh.

- [ ] **Step 7: Add deterministic fallback and inter-node tests**

In `test_ucp_proto_mock.cc`, extend `test_ucp_proto_mock_cuda_ipc` with a tag
lookup helper. Add one case with the host force enabled and a reachable CUDA
IPC child; assert `frag cuda` and `cuda_ipc`. Add a second endpoint configuration
without CUDA IPC and assert that lookup succeeds through a non-CUDA fallback.
Add a predicate-level inter-node case in `test_ucp_proto.cc` with and without
`UCS_MEM_FLAG_MEMTYPE_COPY_INTER_NODE`; only the exportable case may promote
the CUDA IPC child. Reuse existing UCT CUDA IPC tests for the lower-level MNNVL
address/reachability contract rather than duplicating it in UCP.

- [ ] **Step 8: Reserve sender progress in finite host-to-host pools**

For a finite host-to-host CUDA fragment pool larger than one element, limit RTR
allocations to `max_elems - 1` while using the existing mpool and flow-control
queues, leaving one descriptor available to a sender PUT child. Pair the
reservation accounting on every completion, reset, cancellation, and abort
path. With a one-element cap, do not select the host-to-host CUDA-staged
candidate; normal fallback is required. Do not apply this reservation to
CUDA-to-host or to the existing CUDA-via-host policy.

- [ ] **Step 9: Run focused and flow-control tests and verify GREEN**

Run:

```sh
make -C build/test/gtest -j$(nproc) gtest
build/test/gtest/gtest --gtest_filter='*host_cuda*:*test_ucp_proto_mock_cuda_ipc*:*rndv_mtype_fc*:*rndv*frag*abort*'
```

Expected: all new selection, functional, fallback, inter-node eligibility, and
existing flow-control tests pass. Supported functional cases validate data,
observe throttling, and leave both workers' queues empty.

- [ ] **Step 10: Commit Task 2**

```sh
git add src/ucp/rndv/proto_rndv.c src/ucp/rndv/proto_rndv.inl \
        src/ucp/rndv/rndv_mtype.inl src/ucp/rndv/rndv_ppln.c \
        src/ucp/rndv/rndv_rtr.c \
        src/ucp/rndv/rndv_put.c test/gtest/ucp/test_ucp_proto.cc \
        test/gtest/ucp/test_ucp_proto_mock.cc
git commit -m "UCP/RNDV: Force host transfers through CUDA fragments"
```

---

### Task 3: Document the Perftest Enforcement Recipe

**Files:**
- Modify: `docs/source/faq.md`
- Modify: `NEWS`

**Interfaces:**
- Consumes: existing `ucx_perftest -m SEND,RECV`, `UCX_PROTO_INFO`, UCP configuration from Tasks 1-2, and CUDA IPC `ENABLE_MNNVL` configuration.
- Produces: runnable same-node and MNNVL examples plus release-note visibility.

- [ ] **Step 1: Add the perftest example**

Add a short FAQ subsection under protocol selection containing this server
environment and command:

```sh
UCX_TLS=rc,cuda_copy,cuda_ipc \
UCX_PROTO_ENABLE=y \
UCX_PROTO_INFO=y \
UCX_RNDV_THRESH=0 \
UCX_RNDV_PIPELINE_HOST_CUDA_STAGING_FORCE=y \
UCX_RNDV_FRAG_MEM_TYPES=cuda \
UCX_RNDV_FRAG_WORKER_MAX_MEM=256M \
./src/tools/perf/ucx_perftest -t tag_bw -m host,host -s 8388608
```

Show the client form with the server hostname and state that `host,cuda` and
`cuda,host` exercise the asymmetric directions. Explain that
`UCX_PROTO_INFO=y` must show `frag cuda` and `cuda_ipc`; otherwise UCX used the
documented fallback. For cross-node use, state that both peers require CUDA
IPC MNNVL/fabric support and a configured NVIDIA IMEX channel; do not claim
that the UCP option enables MNNVL by itself.

- [ ] **Step 2: Add a NEWS entry**

Add one concise item under the current release's UCP features:

```text
* Added an opt-in tag rendezvous policy to stage host transfers through CUDA
  fragments and CUDA IPC, including MNNVL-capable systems.
```

- [ ] **Step 3: Validate documentation and configuration output**

Run:

```sh
build/src/tools/info/ucx_info -f | \
    rg 'RNDV_PIPELINE_HOST_CUDA_STAGING_FORCE|RNDV_FRAG_WORKER_MAX_MEM'
git diff --check
```

Expected: both configuration variables are printed and `git diff --check`
reports no errors. If the documentation target is configured, also run
`make -C build/docs html` and require success.

- [ ] **Step 4: Commit Task 3**

```sh
git add docs/source/faq.md NEWS
git commit -m "DOCS: Add host CUDA staging perftest example"
```

---

### Task 4: Full Verification and Patch Review

**Files:**
- Review: all files changed since design commit `092227181`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: build/test evidence and a reviewable patch within UCX scope limits.

- [ ] **Step 1: Build the development configuration**

If `build-devel/Makefile` does not exist, run:

```sh
./autogen.sh
mkdir -p build-devel
cd build-devel
../contrib/configure-devel --prefix=$PWD/install
```

Then run:

```sh
make -C build-devel -j$(nproc)
```

Expected: build succeeds with CUDA enabled when the local dependencies are
available. If CUDA is unavailable, record that limitation and still build the
non-CUDA configuration.

- [ ] **Step 2: Run focused tests in the development build**

Run:

```sh
build-devel/test/gtest/gtest \
  --gtest_filter='*host_cuda*:*rndv_force_cuda*:*rndv_mtype_fc*:*rndv*frag*abort*'
```

Expected: PASS; hardware-dependent tests may skip with explicit capability
messages.

- [ ] **Step 3: Run the UCP gtest target**

Run:

```sh
make -C build-devel/test/gtest test TESTS='gtest' \
    GTEST_FILTER='test_ucp_*'
```

Expected: UCP tests pass, subject to documented pre-existing environmental
failures.

- [ ] **Step 4: Review scope, style, and line count**

Run:

```sh
git diff --check 092227181..HEAD
git diff --stat 092227181..HEAD
git diff --numstat 092227181..HEAD
git status --short
```

Confirm fewer than 500 added implementation lines, no generated output is
tracked, the user's pre-existing untracked `build/` remains untouched, and no
public structure or wire header changed.

- [ ] **Step 5: Inspect the final protocol with perftest where hardware permits**

Run the documented host-to-host command against a second process, then repeat
with `-m host,cuda` and `-m cuda,host`. Expected protocol output contains
`frag cuda` and `cuda_ipc`; data validation completes without error. If no
two-process CUDA IPC or MNNVL system is available, report this as an explicit
manual verification gap.

- [ ] **Step 6: Commit any verification-only corrections**

If verification required code changes, repeat the affected RED/GREEN test and
commit only those corrections:

```sh
git add -u src/ucp/rndv test/gtest/ucp docs/source/faq.md NEWS
git commit -m "UCP/RNDV: Fix host CUDA staging verification"
```
