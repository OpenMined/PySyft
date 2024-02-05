# stdlib
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
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
        # TODO: should provide token or username/password but not both
        if bool(username and password) == bool(access_token):
            raise ValueError(
                "Either username and password or access_token must be provided, but not both"
            )

        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.access_token = access_token

    def _create_oauth2_string(self) -> str:
        auth_string = f"user={self.username}\1auth=Bearer {self.access_token}\1\1"
        return base64.b64encode(auth_string.encode("ascii")).decode("ascii")

    def send(self, subject: str, body: str, to: str, from_addr: str) -> None:
        if not subject or not body or not to:
            raise ValueError("Subject, body, and recipient email(s) are required")

        msg = MIMEMultipart()
        msg["From"] = from_addr
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                if server.has_extn("STARTTLS"):
                    server.starttls()
                    server.ehlo()

                if self.access_token:
                    auth_string = self._create_oauth2_string()
                    server.docmd("AUTH", "XOAUTH2 " + auth_string)
                elif self.username and self.password:
                    server.login(self.username, self.password)

                text = msg.as_string()
                server.sendmail(from_addr, to, text)
                print("Email sent!")
        except Exception as e:
            print(f"Failed to send email: {e}")
            raise e
