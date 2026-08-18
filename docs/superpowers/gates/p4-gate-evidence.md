# P4 phase gate — observability and cross-linking

## Step 1-2: Kubernetes Dashboard (single-cluster)

Bring-up `make local-setup`: **REAL EXIT: 0**, both dashboard pods `1/1`.

```
curl -sk https://localhost:8443            -> 200
token (960 chars) -> /api/v1/pod/active-fed -> listed 11 pods
```

Verified by **authorisation**, not by the token being non-empty: a token that
exists but grants nothing would pass a length check and fail a user.

## Step 3: cross-links resolve — both directions

```
MLflow round_0 / round_1: 5/5 tags each
  kfp_run_id           = 0f7a4edd-cc80-44a7-baec-f56304bbe4e4
  kfp_run_url          = http://localhost:8080/#/runs/details/0f7a4edd-...
  temporal_workflow_id = train-714f9549-r0 / -r1
  temporal_workflow_url= http://localhost:8233/namespaces/default/workflows/...
  topology             = single

Temporal memo kfp_run_id = 0f7a4edd-cc80-44a7-baec-f56304bbe4e4   (reverse link)
```

Every id checked against the API that owns it, each with a bogus-id control
proving the check discriminates:

```
KFP  /apis/v2beta1/runs/<tagged id>  -> that run, SUCCEEDED
KFP  /apis/v2beta1/runs/0000...      -> ResourceNotFoundError
Temporal /api/v1/.../workflows/<id>  -> TrainRoundWorkflow, COMPLETED
Temporal /api/v1/.../does-not-exist  -> 404
```

**Fetching the URLs themselves proves nothing** — both are client-side SPA
routes that return 200 for any id, including garbage. Only the backing APIs
distinguish a working link from a plausible one.

## Step 4: multi-cluster dashboards

Bring-up `make multi-setup`: **REAL EXIT: 0**.

```
curl -sk https://localhost:8443  -> 200   (Kubernetes Dashboard)
curl -s  http://localhost:32000  -> 200   (Karmada Dashboard, library component)
clusters: active-fed-host / member1 / member2 all READY=True
```

Karmada token verified as the **sole** credential (no kubeconfig certs), which
is what the dashboard does at login:

```
kubectl --server https://127.0.0.1:32443 --token <token> get clusters
  -> all three clusters listed
kubectl --server ... --token not-a-real-token get clusters
  -> error: You must be logged in to the server (Unauthorized)
```

## Defects this gate found

Both were in the cross-links, and both would have passed a looser check.

1. **`kfp_run_url` pointed at `run_uid`, not KFP's run id.** `run_uid` names
   the MinIO bucket and worker Jobs; KFP's UI knows nothing about it. The
   plan's suggested alternative (`dsl.PIPELINE_JOB_ID_PLACEHOLDER`) does not
   resolve in this deployment either — P2's F4 finding. Fixed by recovering
   KFP's own id from the URI of an artifact it minted (`d9237b4`).

2. **The Temporal memo carried `run_uid` too** — the same bug in the reverse
   direction, surviving the first fix. `memo keys: ['kfp_run_id']` was true
   the entire time the value was `"23859aa5"` against a real run id of
   `3b66067f-...`. Only decoding the value exposed it. Fixed with a separate
   `RoundSpec.kfp_backend_run_id` (`21137a7`), because `kfp_run_id` must keep
   holding `run_uid`: `job_name_for` builds Kubernetes Job names from it and a
   full UUID would exceed the 63-character limit.

The lesson is the same in both: a tag whose key exists, and a URL that is
well-formed, are not evidence that either resolves.

## Step 5: fed-twin multi-cluster with the dashboard supplied by the library

Bring-up `make multi-cluster-setup`: **REAL EXIT: 0**
Run: KFP `federated-twin-multi-cluster-pipeline-s7v6f`, **Succeeded** in 83s

fed-twin's 47 inlined lines (dashboard install, kubeconfig secret, NodePort
patch, token generation) are gone, replaced by the fed-infra component.

The risk this carried was a **silent capability regression**: the dashboard
would deploy perfectly and simply be unloggable-into, because
`fed_dashboard_token` prints to stdout only (so it cannot leak into logs) and
`fed-infra-up` does not surface it. Verified end to end rather than assumed:

```
Generating Karmada Dashboard Access Token
[fed-infra] created 24h dashboard token for karmada-admin-sa in karmada-system (not logged)
Dashboard Access Token (expires in 24h):
  -> 1143 chars printed
```

Both halves hold: the redacted log line proves the token travelled by stdout,
and the token itself reached the user. Used as the **sole** credential it
authenticates:

```
kubectl --server <karmada> --token <token from the run output> get clusters
  multi-cluster-host / member1 / member2   all READY=True
```

Pipeline output, 10 metric rows (not the header-only shape a failed scrape
produces):

```
round,twin_id,mode,reward,loss
1,train-twin-2,TRAIN,24.50,-0.0077
... 10 rows across 2 rounds, 2 train twins and the eval twin
```

## Gate result

All five steps pass. Both dashboards deploy as fed-infra components in both
profiles; both tokens authenticate and authorise; every MLflow run carries all
five cross-link tags and both directions resolve against the APIs that own
them; fed-twin's multi-cluster path works with the library-supplied dashboard
and still prints its token; and fed-infra's `make check` is green (198 tests,
exit 0) with goldens unchanged.
