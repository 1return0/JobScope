"""Windows-to-VM fixed Docker worker over SSH; no Docker TCP API or host mounts."""
from pathlib import Path
import shlex
import time
from uuid import uuid4

from dotenv import dotenv_values

from app.application.document_workspace import ReportUnavailableError
from app.infrastructure.report_export_policy import docker_worker_command


def export_over_ssh(raw: bytes, *, config_file: str, image: str) -> bytes:
    import paramiko

    settings=dotenv_values(config_file)
    host=settings.get("VM_SSH_HOST")
    user=settings.get("VM_SSH_USER")
    password=settings.get("VM_SSH_PASSWORD") or None
    key=settings.get("VM_SSH_KEY_FILE") or None
    if not host or not user or not (password or key):
        raise ReportUnavailableError("report-export-ssh-authentication-not-configured")
    client=paramiko.SSHClient()
    name="agent-report-export-"+uuid4().hex
    started=False
    try:
        # Unknown/mismatched host keys are never automatically accepted.
        known_hosts=Path.home()/".ssh"/"known_hosts"
        if not known_hosts.is_file():
            raise ReportUnavailableError("report-export-ssh-known-host-required")
        client.load_system_host_keys(str(known_hosts))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(host,port=int(settings.get("VM_SSH_PORT") or "22"),username=user,
            password=password,key_filename=key,allow_agent=False,look_for_keys=False,
            timeout=5,auth_timeout=5,banner_timeout=5)
        # Only fixed Docker arguments plus server-generated name/trusted image.
        # Document text is binary stdin, never interpolated into a shell command.
        stdin,stdout,stderr=client.exec_command(shlex.join(docker_worker_command(name,image)),timeout=15)
        started=True
        channel=stdout.channel
        channel.sendall(raw)
        channel.shutdown_write()
        output=bytearray()
        deadline=time.monotonic()+15
        while True:
            if time.monotonic()>deadline:
                raise ReportUnavailableError("report-export-timeout")
            if channel.recv_ready():
                output.extend(channel.recv(65536))
                if len(output)>2_000_000:
                    raise ReportUnavailableError("report-export-output-too-large")
            if channel.recv_stderr_ready():
                channel.recv_stderr(65536)  # Drain, never return/log remote stderr.
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            time.sleep(.01)
        if channel.recv_exit_status()!=0 or not output.startswith(b"PK"):
            raise ReportUnavailableError("report-export-worker-failed")
        return bytes(output)
    except ReportUnavailableError:
        if started:
            try:
                client.exec_command(shlex.join(["docker","rm","-f",name]),timeout=5)
            except Exception:
                pass
        raise
    except (paramiko.SSHException,OSError,ValueError) as error:
        raise ReportUnavailableError("report-export-ssh-unavailable") from error
    finally:
        client.close()
