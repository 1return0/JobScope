"""Run on the VM after building the bundle; no production DB or model access."""
import hashlib
import io
import json
import subprocess
import uuid
import zipfile

try:
    from report_export_policy import docker_worker_command  # Standalone VM bundle.
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from app.infrastructure.report_export_policy import docker_worker_command


def main():
    image="agent-report-export:local"
    payload={"revision":1,"content":{"title":"Sandbox export verification",
        "sections":[{"heading":"Evidence","body":"A public guide is cited.",
            "citations":[{"source_reference":"https://example.edu/guide",
                "evidence_id":"probe","quoted_text":"Public guide"}]}],"user_notes":""}}
    command=docker_worker_command("agent-report-probe-"+uuid.uuid4().hex,image)
    result=subprocess.run(command,input=json.dumps(payload).encode(),capture_output=True,
                          timeout=30,check=True)
    with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
        assert b"Sandbox export verification" in archive.read("word/document.xml")
    # Probe the very same container restrictions, changing only its fixed entrypoint.
    command=docker_worker_command("agent-report-probe-"+uuid.uuid4().hex,image)
    command[-1:-1]=["--entrypoint=python"]
    probe="""
import os,socket,json
assert os.getuid()==65534
root_write_blocked=False
try:
    open('/root-write-probe','w').close()
except OSError:
    root_write_blocked=True
assert root_write_blocked
network_blocked=False
s=socket.socket();s.settimeout(1)
try:
    s.connect(('198.51.100.1',443))
except OSError:
    network_blocked=True
finally:
    s.close()
assert network_blocked
status=open('/proc/self/status').read()
assert 'NoNewPrivs:\\t1' in status
assert int(next(line.split()[1] for line in status.splitlines() if line.startswith('CapEff:')),16)==0
assert 'Max processes' in open('/proc/self/limits').read()
print(json.dumps({'uid':os.getuid(),'root_write_blocked':root_write_blocked,
    'network_blocked':network_blocked,'no_new_privileges':True,'effective_capabilities':0}))
"""
    result2=subprocess.run(command+["-I","-c",probe],capture_output=True,timeout=15,check=True)
    print(json.dumps({"status":"passed","docx_bytes":len(result.stdout),
        "docx_sha256":hashlib.sha256(result.stdout).hexdigest(),
        "isolation":json.loads(result2.stdout),
        "boundary":"Fixed worker export and container restrictions; not production load/escape certification"},indent=2))


if __name__=="__main__":
    main()
