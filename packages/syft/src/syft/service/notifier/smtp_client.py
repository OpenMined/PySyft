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
        if not (values.get("username", None) and values.get("password")):
            raise ValueError("Both username and password must be provided")
        return values

    def send(self, sender: str, receiver: list[str], subject: str, body: str) -> None:
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
        """Check if the credentials are valid.

        Returns:
            bool: True if the credentials are valid, False otherwise.
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
