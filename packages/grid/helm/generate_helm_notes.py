# stdlib
import os
import sys


def add_notes(helm_chart_template_dir: str) -> None:
    """Add notes or information post helm install or upgrade."""

    notes = """
    Thank you for installing {{ .Chart.Name }}.
    Your release is named {{ .Release.Name }}.
    To learn more about the release, try:

        $ helm status {{ .Release.Name }} -n {{ .Release.Namespace }}
        $ helm get all {{ .Release.Name }}

    =========================================
    Syft Installed
    =========================================
    Check the ingress for your IP by running the command below:

        kubectl get ingress -n {{ .Release.Namespace }}

    Once you have the IP, visit it in your browser:

        http://<IP>

    You should see the welcome page. You can now log in with Syft in Python or Jupyter:

    import syft as sy
    sy.login(url="http://<IP>", email="info@openmined.org", password="yourpass")

    You can see your password in the configmap with:

        kubectl get secret backend-secret -n syft -o yaml | grep defaultRootPassword

    If you used a different email or password, make sure to adjust the login information accordingly.

    =========================================
    Start Jupyter Environment
    =========================================
    You can start a Jupyter environment using the following Docker command:

        docker run --rm -it --network=host openmined/syft-client:${VERSION}

    Consider using `tmux` to keep the Jupyter notebook running in the background.
    This can help manage long-running sessions and maintain your environment active.

    For more information on installation and usage, refer to the OpenMined documentation:

    https://docs.openmined.org

    =========================================
    """

    notes_path = os.path.join(helm_chart_template_dir, "NOTES.txt")

    with open(notes_path, "w") as fp:
        fp.write(notes)


if __name__ == "__main__":
    # Write code to path from user and pass to generate notes
    if len(sys.argv) != 2:
        print("Please provide helm chart template directory path")
        sys.exit(1)
    helm_chart_template_dir = sys.argv[1]
    add_notes(helm_chart_template_dir)
    print("=" * 50)
    print("Notes Generated Successfully")
    print("=" * 50)
