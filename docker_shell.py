import io
import tarfile
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
        self._python_session_started = False

    def start(self):
        """Start Docker container"""
        self.container = self.client.containers.run(
            self.image,
            command="tail -f /dev/null",
            detach=True,
            # volumes={str(self.workspace_dir.absolute()): {'bind': '/workspace', 'mode': 'rw'}},
            # working_dir='/workspace',
            mem_limit='2g',
            remove=True,
            stdin_open=True,
            tty=True
        )

    def _start_python_session(self):
        """Start a persistent Python session in the container"""
        if not self.container:
            raise RuntimeError("Container not started")

        # Create a named pipe for Python session
        self.exec("mkfifo /tmp/python_input")

        # Start Python in background reading from the pipe
        self.exec("""
nohup bash -c '
python3 -u << "END_PYTHON" > /tmp/python_output 2>&1 &
import sys
import traceback

while True:
    try:
        with open("/tmp/python_input", "r") as f:
            code = f.read()

        if code.strip() == "__EXIT__":
            break

        print("---EXEC_START---")
        sys.stdout.flush()

        try:
            exec(code, globals())
            print("---EXEC_SUCCESS---")
        except Exception as e:
            print("---EXEC_ERROR---")
            traceback.print_exc()

        sys.stdout.flush()
    except Exception as e:
        traceback.print_exc()
        break
END_PYTHON
' > /dev/null 2>&1 &
""")

        self._python_session_started = True

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
        """Execute Python code in persistent session"""
        if not self._python_session_started:
            self._start_python_session()
            # Give Python time to start
            import time
            time.sleep(0.5)

        # Write code to the input pipe
        escaped_code = code.replace("'", "'\\''")
        write_cmd = f"echo '{escaped_code}' > /tmp/python_input"
        self.exec(write_cmd)

        # Wait for execution and read output
        import time
        max_wait = timeout
        waited = 0

        while waited < max_wait:
            result = self.exec("cat /tmp/python_output 2>/dev/null || echo ''")
            output = result['stdout']

            if '---EXEC_SUCCESS---' in output or '---EXEC_ERROR---' in output:
                # Parse output
                lines = output.split('\n')
                start_idx = None
                end_idx = None

                for i, line in enumerate(lines):
                    if '---EXEC_START---' in line:
                        start_idx = i + 1
                    elif '---EXEC_SUCCESS---' in line or '---EXEC_ERROR---' in line:
                        end_idx = i
                        break

                if start_idx is not None and end_idx is not None:
                    actual_output = '\n'.join(lines[start_idx:end_idx])
                    success = '---EXEC_SUCCESS---' in output

                    # Clear output file for next execution
                    self.exec("truncate -s 0 /tmp/python_output")

                    return {
                        'exit_code': 0 if success else 1,
                        'stdout': actual_output,
                        'stderr': '' if success else actual_output,
                        'success': success
                    }

            time.sleep(0.1)
            waited += 0.1

        return {
            'exit_code': 1,
            'stdout': '',
            'stderr': 'Timeout waiting for Python execution',
            'success': False
        }

    def create_file(self, filename: str, content: str) -> Dict:
        """Create a file with content"""
        command = f"""
cat > {filename} << 'ENDOFFILE'
{content}
ENDOFFILE
"""
        return self.exec(command)

    def get_file(self, container_path: str) -> bytes:
        """
        Get file content from container as bytes.

        Args:
            container_path: Path to file inside container (e.g., '/tmp/output.txt')

        Returns:
            File content as bytes

        Raises:
            RuntimeError: If container not started
            FileNotFoundError: If file doesn't exist in container
        """
        if not self.container:
            raise RuntimeError("Container not started")

        # Check if file exists
        result = self.exec(f"test -f {container_path} && echo 'exists'")
        if 'exists' not in result['stdout']:
            raise FileNotFoundError(f"File not found in container: {container_path}")

        # Get file as tar archive
        bits, stat = self.container.get_archive(container_path)

        # Extract file from tar
        file_obj = io.BytesIO()
        for chunk in bits:
            file_obj.write(chunk)
        file_obj.seek(0)

        # Open tar and extract content
        with tarfile.open(fileobj=file_obj) as tar:
            member = tar.getmembers()[0]
            file_content = tar.extractfile(member).read()

        return file_content

    def stop(self):
        """Stop container and Python session"""
        if self._python_session_started:
            # Signal Python to exit
            self.exec("echo '__EXIT__' > /tmp/python_input")

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
    shell = IsolatedShell(image='python:3.10-slim')

    with shell as sh:
        # Test persistent session - variables should persist
        print("=== Test 1: Setting variable ===")
        result = sh.exec_python("x = 42")
        print(result)

        print("\n=== Test 2: Using previously set variable ===")
        result = sh.exec_python("print(f'x = {x}')")
        print(result)

        print("\n=== Test 3: File creation ===")
        code = """
with open('hello_world.txt', 'w') as file: 
    file.write('Hello, World!')
print('File created')
"""
        result = sh.exec_python(code)
        print(result)

        print("\n=== Test 4: List files ===")
        res = sh.exec('ls -la')
        print(res)