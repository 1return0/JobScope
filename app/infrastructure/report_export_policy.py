"""One fixed Docker policy used by the runtime and independent isolation probe."""
def docker_worker_command(name, image):
    return ["docker","run","--rm","--name",name,"--network=none",
        "--read-only","--cap-drop=ALL","--security-opt=no-new-privileges",
        "--user=65534:65534","--memory=128m","--memory-swap=128m",
        "--cpus=0.5","--pids-limit=32","--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
        "-i",image]
