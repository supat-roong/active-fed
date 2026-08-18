# P3 phase gate — live multi-cluster run

Run: KFP `active-fl-cartpole-w8llk`, **Succeeded** in 4m26s
Config: `config/k8s-multi.yaml` (topology=multi, members=2, 2 rounds, 2 workers)

## Step 1 — three clusters, all federated

```
$ kind get clusters
active-fed-host / active-fed-member1 / active-fed-member2

$ kubectl --kubeconfig ~/.karmada/... get clusters
NAME                 VERSION   MODE   READY
active-fed-host      v1.35.0   Push   True
active-fed-member1   v1.35.0   Push   True
active-fed-member2   v1.35.0   Push   True
```

## Step 2 — workers land on separate members, one each

```
member1   aflw-6a00c6f0-r0-w0-rscxq   Running
member2   aflw-6a00c6f0-r0-w1-czv8k   Running

aflw-...-r0-w0-pp: selects Job/aflw-...-r0-w0 -> ['active-fed-member1']
aflw-...-r0-w1-pp: selects Job/aflw-...-r0-w1 -> ['active-fed-member2']
```

`worker_id % member_count + 1` in `RoundSpec.worker_spec`, live. Each policy
pins exactly one cluster -- never an empty `clusterNames`, which would mean
*all* clusters.

## Step 3 — members genuinely separate

Worker pods on the host cluster: **none**, on every check during both rounds.

## Step 4 — MinIO completion detection, no cross-cluster Job watching

Both rounds advanced to completion with the host never watching a member's
Job. Bucket `fed-2907a8d8640b`:

```
round_0/global.pt
round_0/workers/worker_{0,1}_{delta.pt,metrics.json,weights.pt}
round_1/global.pt
round_1/workers/worker_{0,1}_{delta.pt,metrics.json,weights.pt}
```

Both workers' updates collected in both rounds, and `round_1/global.pt`
proves the aggregator ran on round 0's collected results.

Karmada secrets in the `active-fed` namespace: only `karmada-kubeconfig`,
the one mounted into the Temporal worker.

## Step 5 — MLflow records the rounds

5 runs, all FINISHED, logged from workers on member clusters:
`worker_0_round_0`, `worker_1_round_0`, `worker_0_round_1`,
`worker_1_round_1`, `round_0` (aggregation).

## Defects this gate found that no unit test could

1. **`404 namespaces "active-fed" not found`** -- the Karmada control plane is
   a separate apiserver with its own namespaces. The test fakes modelled one
   apiserver, so nothing could have caught it. Fixed in `502652d`.
2. **Members cannot resolve `*.svc.cluster.local` of the host** -- a member
   cluster has its own DNS, so MinIO/MLflow were unreachable. Endpoints are
   now rewritten to `<host-node-InternalIP>:<NodePort>` at dispatch time, IP
   resolved dynamically because a kind node's Docker IP changes across a host
   restart (observed: 172.18.0.3 -> 172.18.0.2). Fixed in `43445e6`.
3. **`nodes is forbidden`** -- the RBAC granting that lookup lived in a
   different file with nothing tying it to the code. Fixed in the RBAC commit,
   with a test that fails when the grant is removed.

Findings 2 and 3 were both diagnosed in one step because the endpoint rewrite
was specified to fail loudly rather than fall back to the DNS name: the error
named the exact ServiceAccount and permission instead of timing out.

## Infrastructure defects found (fed-infra)

- Karmada apiserver NodePort 32443 unmapped -> every join would fail
- `fed_karmada_init` treated namespace-exists as initialized, while
  `--etcd-storage-mode=emptyDir` loses all state on restart
- `helm upgrade` collided with the NodePort patch's field manager, so
  `fed-infra-up` was not re-runnable
- inotify limits were *lowered* on re-run (sysctl is shared with the host
  kernel, not namespaced)
- pod-readiness budget was 12 attempts with a comment claiming "~120s"

## Environment note

The Docker disk filled twice (79G, 100%), which presents as slow pulls and
regressing pod counts. `crictl pull` by hand and `df` on the *docker*
filesystem -- not the VM root -- is the fast diagnosis. Grown to 118G.
