#!/usr/bin/env bash
# Fixed isolated exporter probe; no credentials, DB, host mounts or model calls.
set -euo pipefail
image=agent-report-export:local
policy=(--rm --network=none --read-only --cap-drop=ALL
  --security-opt=no-new-privileges --user=65534:65534
  --memory=128m --memory-swap=128m --cpus=0.5 --pids-limit=32
  --tmpfs=/tmp:rw,noexec,nosuid,size=16m -i)
payload='{"revision":1,"content":{"title":"Sandbox verification","sections":[{"heading":"Evidence","body":"Public guide","citations":[{"source_reference":"https://example.edu/guide","evidence_id":"probe"}]}],"user_notes":""}}'
printf '%s' "$payload" | docker run "${policy[@]}" "$image" > export-probe.docx
docker run "${policy[@]}" --entrypoint=python "$image" -I -c '
import sys,io,zipfile,hashlib,json
data=sys.stdin.buffer.read()
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    assert b"Sandbox verification" in archive.read("word/document.xml")
print(json.dumps({"export":"passed","bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}))
' < export-probe.docx
docker run "${policy[@]}" --entrypoint=python "$image" -I -c '
import os,socket,json
assert os.getuid()==65534
blocked=False
try:
    open("/root-write-probe","w").close()
except OSError:
    blocked=True
assert blocked
with open("/tmp/scratch-probe","w") as scratch:
    scratch.write("temporary only")
network_blocked=False
s=socket.socket();s.settimeout(1)
try:
    s.connect(("198.51.100.1",443))
except OSError:
    network_blocked=True
finally:
    s.close()
assert network_blocked
status=open("/proc/self/status").read()
assert "NoNewPrivs:\t1" in status
assert int(next(line.split()[1] for line in status.splitlines() if line.startswith("CapEff:")),16)==0
print(json.dumps({"isolation":"passed","uid":os.getuid(),"root_write_blocked":blocked,"network_blocked":network_blocked,"no_new_privileges":True,"effective_capabilities":0}))
'
