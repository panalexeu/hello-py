import tempfile
from pathlib import Path
from typing import Dict, Optional

import docker
from rich import print


class IsolatedShell:
    def __init__(self, workspace_dir: Optional[str] = None):
        self.client = docker.from_env()
        self.workspace_dir = Path(workspace_dir or tempfile.mkdtemp())
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        self.container = None

    def start(self):
        """Start Docker container"""
        self.container = self.client.containers.run(
            "python:3.10-slim",
            command="tail -f /dev/null",
            detach=True,
            volumes={str(self.workspace_dir.absolute()): {'bind': '/workspace', 'mode': 'rw'}},
            working_dir='/workspace',
            mem_limit='2g',
            remove=True
        )

    def exec(self, command: str, timeout: int = 30) -> Dict:
        """Execute command in container"""
        if not self.container:
            raise RuntimeError("Container not started")

        exit_code, output = self.container.exec_run(
            ["bash", "-c", command],  # Pass as list instead of string
            demux=True
        )

        return {
            'exit_code': exit_code,
            'stdout': output[0].decode('utf-8') if output[0] else "",
            'stderr': output[1].decode('utf-8') if output[1] else "",
            'success': exit_code == 0
        }

    def stop(self):
        """Stop container"""
        if self.container:
            self.container.stop()


# test the IsolatedShell works
if __name__ == '__main__':
    shell = IsolatedShell()

    shell.start()

    # run python code that creates `hello_world.txt`, then examine the output with `cat`:
    code = """
with open('hello_world.txt', 'w') as file: 
    file.write('Hello, World!')
    """
    shell.exec(f"cat > script.py << 'EOF'\n{code}\nEOF")  # save multiline code
    shell.exec("python ./script.py")
    res = shell.exec('cat ./hello_world.txt')

    shell.stop()

    print(res)

