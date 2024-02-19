# stdlib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from typing import List
from typing import Optional


class SMTPClient:
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        # Should provide token or username/password but not both
        if username and password and access_token:
            raise ValueError(
                "Either username and password or access_token must be provided, but not both"
            )

        if not (username and password) and not access_token:
            raise ValueError(
                "Either username and password or access_token must be provided"
            )

        if username and password:
            self.username = username
            self.password = password
        else:
            self.access_token = access_token

        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send(self, sender: str, receiver: List[str], subject: str, body: str) -> None:
        if not subject or not body or not receiver:
            raise ValueError("Subject, body, and recipient email(s) are required")

        msg = MIMEMultipart("alternative")
        msg["From"] = sender
        msg["To"] = ", ".join(receiver)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.ehlo()
            if server.has_extn("STARTTLS"):
                server.starttls()
                server.ehlo()

            if self.access_token:
                server.login(self.access_token, self.access_token)
            elif self.username and self.password:
                server.login(self.username, self.password)

            text = msg.as_string()
            server.sendmail(sender, ", ".join(receiver), text)
