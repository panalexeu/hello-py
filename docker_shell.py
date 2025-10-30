import tempfile
from pathlib import Path
from typing import Dict, Optional

import docker
from rich import print


class IsolatedShell:
    def __init__(
            self,
            workspace_dir: Optional[str] = None,
            image: str = "python:3.10-slim",
            auto_install: bool = True
    ):
        self.client = docker.from_env()
        self.workspace_dir = Path(workspace_dir or tempfile.mkdtemp())
        self.workspace_dir.mkdir(exist_ok=True, parents=True)
        self.image = image
        self.auto_install = auto_install
        self.container = None

    def start(self):
        """Start Docker container"""
        self.container = self.client.containers.run(
            self.image,
            command="tail -f /dev/null",
            detach=True,
            # volumes={str(self.workspace_dir.absolute()): {'bind': '/workspace', 'mode': 'rw'}},
            # working_dir='/workspace',
            mem_limit='2g',
            remove=True
        )

    def exec(self, command: str, timeout: int = 30) -> Dict:
        """Execute command in container"""
        if not self.container:
            raise RuntimeError("Container not started")

        exit_code, output = self.container.exec_run(
            ["bash", "-c", command],
            demux=True
        )

        return {
            'exit_code': exit_code,
            'stdout': output[0].decode('utf-8') if output[0] else "",
            'stderr': output[1].decode('utf-8') if output[1] else "",
            'success': exit_code == 0
        }

    def exec_python(self, code: str, timeout: int = 30) -> Dict:
        """Execute multiline Python code"""
        command = f"""
cat > /tmp/temp_script.py << 'ENDOFPYTHON'
{code}
ENDOFPYTHON
python /tmp/temp_script.py
"""
        return self.exec(command, timeout)

    def create_file(self, filename: str, content: str) -> Dict:
        """Create a file with content"""
        command = f"""
cat > {filename} << 'ENDOFFILE'
{content}
ENDOFFILE
"""
        return self.exec(command)

    def stop(self):
        """Stop container"""
        if self.container:
            self.container.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# test the IsolatedShell works
if __name__ == '__main__':
    # before running the code build the container with: docker build -t ml-env .
    shell = IsolatedShell(image='ml-env')

    with shell as sh:
        # run python code that creates `hello_world.txt`, then examine the output with `cat`:
        code = """
with open('hello_world.txt', 'w') as file: 
    file.write('Hello, World!')
        """
        shell.exec_python(code)
        shell.exec("python ./script.py")
        res = shell.exec('ls -la')
        print(res)
