"""Test the docker wrapper utility."""

import pytest
from langchain.utilities.docker import DockerWrapper, \
            gvisor_runtime_available, _default_params
from unittest.mock import MagicMock
import subprocess


def docker_installed() -> bool:
    """Checks if docker is installed locally."""
    try:
        subprocess.run(['which', 'docker',], check=True)
    except subprocess.CalledProcessError:
        return False

    return True




@pytest.mark.skipif(not docker_installed(), reason="docker not installed")
class TestDockerUtility:

    def test_default_image(self) -> None:
        """Test running a command with the default alpine image."""
        docker = DockerWrapper()
        output = docker.run('cat /etc/os-release')
        assert output.find('alpine')

    def test_shell_escaping(self) -> None:
        docker = DockerWrapper()
        output = docker.run('echo "hello world" | sed "s/world/you/g"')
        assert output == 'hello you'
        # using embedded quotes
        output = docker.run("echo 'hello world' | awk '{print $2}'")
        assert output == 'world'

    def test_auto_pull_image(self) -> None:
        docker = DockerWrapper(image='golang:1.20')
        output = docker.run("go version")
        assert output.find('go1.20')
        docker._docker_client.images.remove('golang:1.20')

    def test_inner_failing_command(self) -> None:
        """Test inner command with non zero exit"""
        docker = DockerWrapper()
        output = docker.run('ls /inner-failing-command')
        assert str(output).startswith("STDERR")

    def test_entrypoint_failure(self) -> None:
        """Test inner command with non zero exit"""
        docker = DockerWrapper()
        output = docker.run('todo handle APIError')
        assert output == 'ERROR'

    def test_check_gvisor_runtime(self) -> None:
        """test gVisor runtime verification using a mock docker client"""
        mock_client = MagicMock()
        mock_client.info.return_value = {'Runtimes': {'runsc': {'path': 'runsc'}}}
        assert gvisor_runtime_available(mock_client)
        mock_client.info.return_value = {'Runtimes': {'runc': {'path': 'runc'}}}
        assert not gvisor_runtime_available(mock_client)

    def test_socket_read_timeout(self) -> None:
        """Test socket read timeout."""
        docker = DockerWrapper(image='python', command='python')
        # this query should fail as python needs to be started with python3 -i
        output = docker.exec_run("test query")
        assert output == "ERROR: timeout"

def test_get_image_template() -> None:
    """Test getting an image template instance from string."""
    from langchain.utilities.docker.images import get_image_template
    image = get_image_template("python")
    assert image.__name__ == "Python" #  type: ignore

def test_default_params() -> None:
    """Test default container parameters."""
    docker = DockerWrapper(image="my_custom_image")
    assert docker._params == {**_default_params(), "image": "my_custom_image"}
