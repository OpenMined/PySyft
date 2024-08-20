# stdlib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

# third party
from pydantic import BaseModel
from pydantic import model_validator
from result import Err
from result import Ok
from result import Result

SOCKET_TIMEOUT = 5  # seconds


class SMTPClient(BaseModel):
    username: str
    password: str
    server: str
    port: int

    @model_validator(mode="before")
    @classmethod
    def check_user_and_password(cls, values: dict) -> dict:
        """Validate that both username and password are provided."""
        if not (values.get("username", None) and values.get("password")):
            raise ValueError("Both username and password must be provided")
        return values

    def send(self, sender: str, receiver: list[str], subject: str, body: str) -> None:
        """Send an email using the SMTP server.

        Args:
            sender (str): The sender's email address.
            receiver (list[str]): A list of recipient email addresses.
            subject (str): The subject of the email.
            body (str): The HTML body of the email.

        Raises:
            ValueError: If subject, body, or receiver is not provided.
        """
        if not (subject and body and receiver):
            raise ValueError("Subject, body, and recipient email(s) are required")

        msg = MIMEMultipart("alternative")
        msg["From"] = sender
        msg["To"] = ", ".join(receiver)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(self.server, self.port, timeout=SOCKET_TIMEOUT) as server:
            server.ehlo()
            if server.has_extn("STARTTLS"):
                server.starttls()
                server.ehlo()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(sender, ", ".join(receiver), text)
        # TODO: Add error handling

    @classmethod
    def check_credentials(
        cls, server: str, port: int, username: str, password: str
    ) -> Result[Ok, Err]:
        """Check if the provided SMTP credentials are valid.

        Args:
            server (str): The SMTP server address.
            port (int): The port number to connect to.
            username (str): The username for the SMTP server.
            password (str): The password for the SMTP server.

        Returns:
            Result[Ok, Err]: Ok if the credentials are valid, Err with an exception otherwise.
        """
        try:
            with smtplib.SMTP(server, port, timeout=SOCKET_TIMEOUT) as smtp_server:
                smtp_server.ehlo()
                if smtp_server.has_extn("STARTTLS"):
                    smtp_server.starttls()
                    smtp_server.ehlo()
                smtp_server.login(username, password)
                return Ok("Credentials are valid.")
        except Exception as e:
            return Err(e)
