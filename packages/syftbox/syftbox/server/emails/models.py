from typing import Union

from pydantic import BaseModel, EmailStr, NameEmail

FROM_EMAIL = "SyftBox <notifications@syftbox.openmined.org>"


class SendEmailRequest(BaseModel):
    to: Union[EmailStr, NameEmail]
    subject: str
    html: str

    def json_for_request(self) -> dict:
        return {
            "personalizations": [{"to": [{"email": self.to}]}],
            "from": {"email": FROM_EMAIL},
            "subject": self.subject,
            "content": [{"type": "text/html", "value": self.html}],
        }
