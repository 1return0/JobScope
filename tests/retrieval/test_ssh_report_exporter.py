from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.application.document_workspace import ReportUnavailableError
from app.infrastructure.ssh_report_exporter import export_over_ssh


def test_missing_credentials_stop_before_connecting(tmp_path):
    config=tmp_path/"vm.env"
    config.write_text("VM_SSH_HOST=192.168.40.128\nVM_SSH_USER=root\n",encoding="utf8")
    with patch("paramiko.SSHClient") as factory:
        with pytest.raises(ReportUnavailableError,match="authentication-not-configured"):
            export_over_ssh(b"{}",config_file=str(config),image="agent-report-export:local")
        factory.assert_not_called()


def test_ssh_command_contains_only_fixed_policy_document_is_stdin(tmp_path):
    config=tmp_path/"vm.env"
    config.write_text("VM_SSH_HOST=192.168.40.128\nVM_SSH_USER=root\nVM_SSH_PASSWORD=test-password\n",encoding="utf8")
    class Channel:
        data=b"PKdocument"
        sent=None
        def sendall(self,value):self.sent=value
        def shutdown_write(self):pass
        def recv_ready(self):return bool(self.data)
        def recv(self,count):value,self.data=self.data,b"";return value
        def recv_stderr_ready(self):return False
        def exit_status_ready(self):return True
        def recv_exit_status(self):return 0
    channel=Channel()
    with patch("paramiko.SSHClient") as factory,patch.object(Path,"is_file",return_value=True):
        client=factory.return_value
        client.exec_command.return_value=(None,SimpleNamespace(channel=channel),None)
        result=export_over_ssh(b'"document-text-sentinel"',config_file=str(config),image="agent-report-export:local")
        assert result==b"PKdocument"
        assert channel.sent==b'"document-text-sentinel"'
        command=client.exec_command.call_args.args[0]
        assert "--network=none" in command and "--cap-drop=ALL" in command
        assert "document-text-sentinel" not in command and "test-password" not in command
        assert client.connect.call_args.kwargs["look_for_keys"] is False
        client.set_missing_host_key_policy.assert_called_once()
        client.close.assert_called_once()
