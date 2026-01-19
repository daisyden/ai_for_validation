from typing_extensions import Tuple
import docker

def run_in_docker(command: str, container: str, workdir: str) -> str:
    def run_command_in_existing_container(container_id, command):
        """
        Execute a command in an existing running container
        """
        client = docker.from_env()
        
        # Get the container by ID
        container = client.containers.get(container_id)

        # Execute command in the container
        exec_result = container.exec_run(
            command,
            stdout=True,
            stderr=True,
            stream=True,  # Stream output in real-time
            workdir=workdir
        )
        
        output_lines = []
        
        # Stream the output
        if exec_result.output:
            for line in exec_result.output:
                if line:
                    decoded_line = line.decode('utf-8') if isinstance(line, bytes) else line
                    print(f"Output: {decoded_line.strip()}")
                    output_lines.append(decoded_line)
        
        return "\n".join(output_lines)

    # Usage
    output = run_command_in_existing_container(
        container_id=container,  # Your container ID
        command=command  # Command to run
    )

    return output
    


def run_shell_command(command: str) -> str:    
    """
    Execute a shell command and return its output.
    
    Args:
        command: The shell command to execute.
    """
    import subprocess
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"


def run_in_tmux_wait(command: str, tmux_session: str) -> Tuple[str, float]:
    """
    Execute a shell command inside a tmux session, wait for its completion (via tmux wait),
    then capture and return the pane output.
    """ 
    import subprocess
    try:
        # Send the command to the tmux session:
        # 1. clear the screen
        # 2. run the user command
        # 3. signal completion with 'tmux wait -S my_lock'
        import time
        timestamp = time.time()
        full_command = f"tmux clear-history; clear ; {command} ; tmux wait -S my_lock_{timestamp}"
        subprocess.run(
            ["tmux", "send-keys", "-t", tmux_session, full_command, "C-m"],
            check=True
        )

        # Wait until the in-pane command signals it is done
        subprocess.run(
            ["tmux", "wait", f"my_lock_{timestamp}"],
            check=True
        )

        # Capture the entire pane contents
        result = subprocess.run(
            ["tmux", "capture-pane", "-J", "-S", "-", "-E", "-", "-p", "-t", tmux_session],
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip(), timestamp
    except subprocess.CalledProcessError as e:
        return f"Error executing command in tmux: {e}"
    except Exception as e:
        return f"An error occurred: {e}"
