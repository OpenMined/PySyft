import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from syftbox.lib.constants import SENDGRID_API_URL
from syftbox.server.emails.models import SendEmailRequest
from syftbox.server.settings import ServerSettings, get_server_settings

router = APIRouter(prefix="/emails", tags=["email"])


@router.post("/")
async def send_email(
    email_request: SendEmailRequest,
    server_settings: ServerSettings = Depends(get_server_settings),
) -> bool:
    # TODO add some safety mechanisms (rate limiting, authorization, etc)
    if not server_settings.sendgrid_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email service API key is not set",
        )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            SENDGRID_API_URL,
            headers={
                "Authorization": f"Bearer {server_settings.sendgrid_secret.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json=email_request.json_for_request(),
        )
        if response.is_success:
            logger.info(f"Email sent successfully to '{email_request.to}'")
            return True
        else:
            logger.error(f"Failed to send email: {response.text}")
            return False
