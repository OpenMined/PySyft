from typing import Optional

import httpx
from jinja2 import Template
from loguru import logger

from syftbox.lib.constants import SENDGRID_API_URL
from syftbox.server.settings import ServerSettings

SENDER_EMAIL = "SyftBox <auth@syftbox.openmined.org>"

token_email_template = """
<!doctype html>
<html>
  <head>
    <style>
      body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
        color: #333;
        margin: 0;
        padding: 0;
      }
      .container {
        max-width: 600px;
        margin: 50px auto;
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
      h1 {
        color: #333;
        text-align: center;
      }
      p {
        font-size: 16px;
        line-height: 1.5;
      }
      code {
        display: block;
        background-color: #f0f0f0;
        color: #ff8c00;
        padding: 10px;
        font-size: 14px;
        margin: 20px auto;
        border-radius: 4px;
        width: 90%;
        word-wrap: break-word;
      }
      .footer {
        text-align: center;
        font-size: 12px;
        color: #aaa;
        margin-top: 20px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Welcome!</h1>
      <p>
        Use the following token in your CLI to complete your registration:
      </p>
      <code> {{ token }} </code>
      <p>If you did not request this, please ignore this email.</p>
    </div>
  </body>
</html>
"""

reset_password_token_email_template = """
<!doctype html>
<html>
  <head>
    <style>
      body {
        font-family: Arial, sans-serif;
        background-color: #f4f4f4;
        color: #333;
        margin: 0;
        padding: 0;
      }
      .container {
        max-width: 600px;
        margin: 50px auto;
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
      h1 {
        color: #333;
        text-align: center;
      }
      p {
        font-size: 16px;
        line-height: 1.5;
      }
      code {
        display: block;
        background-color: #f0f0f0;
        color: #ff8c00;
        padding: 10px;
        font-size: 14px;
        margin: 20px auto;
        border-radius: 4px;
        width: 90%;
        word-wrap: break-word;
      }
      .footer {
        text-align: center;
        font-size: 12px;
        color: #aaa;
        margin-top: 20px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Hello!</h1>
      <p>
        Use the following command in your CLI to reset your password:
      </p>
      <code> syftbox reset-password --email {{ email }} --token {{ token }} </code>
      <p>If you did not request this, please ignore this email.</p>
    </div>
  </body>
</html>
"""


def send_token_email(server_settings: ServerSettings, user_email: str, token: str) -> None:
    template = Template(token_email_template)
    body = template.render(email=user_email, token=token)
    send_email(
        server_settings=server_settings,
        receiver_email=user_email,
        subject="SyftBox Token",
        body=body,
        mimetype="text/html",
    )


def send_email(
    server_settings: ServerSettings,
    receiver_email: str,
    subject: str,
    body: str,
    mimetype: str = "text/html",
) -> Optional[dict]:
    payload = {
        "personalizations": [{"to": [{"email": receiver_email}]}],
        "from": {"email": SENDER_EMAIL},
        "subject": subject,
        "content": [{"type": mimetype, "value": body}],
    }

    if server_settings.sendgrid_secret is None:
        raise ValueError("Sendgrid secret is not configured")

    headers = {
        "Authorization": f"Bearer {server_settings.sendgrid_secret.get_secret_value()}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(SENDGRID_API_URL, json=payload, headers=headers, timeout=10.0)
        response.raise_for_status()
        logger.info(f"Email sent to {receiver_email}")
        return {"success": True, "status_code": response.status_code}
    except httpx.HTTPError as e:
        logger.error(str(e))
        return None
