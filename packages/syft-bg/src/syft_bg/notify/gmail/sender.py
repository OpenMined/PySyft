"""Gmail email sender."""

import base64
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from syft_bg.notify.email_templates import TemplateRenderer


def _format_interval(seconds: int) -> str:
    """Render an interval as a short human string (e.g. '60s', '5m', '24h')."""
    if seconds % 86400 == 0 and seconds >= 86400:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0 and seconds >= 3600:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0 and seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


@dataclass
class SendResult:
    """Result of sending an email."""

    success: bool
    error_message: str | None = None
    thread_id: str | None = None


class GmailSender:
    """Sends emails via Gmail API."""

    def __init__(self, credentials: Credentials, use_html: bool = True):
        self.service = build("gmail", "v1", credentials=credentials)
        self.use_html = use_html
        self._renderer: Optional[TemplateRenderer] = None

    def verify(self) -> None:
        """Verify credentials by fetching the Gmail user profile."""
        self.service.users().getProfile(userId="me").execute()

    @property
    def renderer(self) -> Optional[TemplateRenderer]:
        if self._renderer is None:
            self._renderer = TemplateRenderer()
        return self._renderer

    def send_email(
        self,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> SendResult:
        """Send an email, optionally in an existing thread."""
        try:
            if body_html and self.use_html:
                message = MIMEMultipart("alternative")
                message["to"] = to_email
                message["subject"] = subject
                part_text = MIMEText(body_text, "plain")
                message.attach(part_text)
                part_html = MIMEText(body_html, "html")
                message.attach(part_html)
            else:
                message = MIMEText(body_text)
                message["to"] = to_email
                message["subject"] = subject

            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
            body_payload: dict = {"raw": raw_message}
            if thread_id:
                body_payload["threadId"] = thread_id

            result = (
                self.service.users()
                .messages()
                .send(userId="me", body=body_payload)
                .execute()
            )
            return SendResult(
                success=True,
                thread_id=result.get("threadId"),
            )
        except Exception:
            import traceback

            print(
                f"[GmailSender] Failed to send to {to_email}: {traceback.format_exc()}"
            )
            return SendResult(success=False, error_message=str(traceback.format_exc()))

    def notify_heartbeat(
        self,
        do_email: str,
        interval_seconds: int,
        timestamp: Optional[datetime] = None,
    ) -> SendResult:
        """Send a heartbeat email confirming the service is still running."""
        subject = "heartbeat: syft-bg still running, token is active"
        ts = timestamp or datetime.now()
        interval_human = _format_interval(interval_seconds)
        body_text = (
            "syft-bg is still running.\n\n"
            f"Account: {do_email}\n"
            f"Sent: {ts:%Y-%m-%d %H:%M:%S}\n"
            f"Heartbeat interval: {interval_human}\n\n"
            "Receiving this email confirms the notify service is up and the "
            "Gmail token is still valid. If these stop arriving, check "
            "`syft-bg status`.\n"
        )
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/heartbeat.html",
                    {
                        "do_email": do_email,
                        "timestamp": ts,
                        "interval_human": interval_human,
                    },
                )
            except Exception:
                pass

        return self.send_email(do_email, subject, body_text, body_html)

    def notify_new_job(
        self,
        do_email: str,
        job_name: str,
        submitter: str,
        timestamp: Optional[datetime] = None,
        job_url: Optional[str] = None,
        job_code: Optional[dict[str, str]] = None,
    ) -> SendResult:
        """Notify DO about a new job."""
        subject = f"Job: {job_name}"
        body_text = f"""You have a new job request in SyftBox!

Job: {job_name}
From: {submitter}
"""
        if job_code:
            body_text += "\nFiles: " + ", ".join(job_code.keys()) + "\n"
            body_text += "\n--- Code ---\n"
            for filename, contents in job_code.items():
                body_text += f"#{filename}\n{contents}\n\n"
            body_text += "--- End Code ---\n"

        body_text += """
To approve or deny this job, reply to this email with:
  approve
  auto-approve
  deny <reason>
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/new_job.html",
                    {
                        "job_name": job_name,
                        "submitter": submitter,
                        "timestamp": timestamp or datetime.now(),
                        "job_url": job_url,
                        "job_code": job_code,
                    },
                )
            except Exception:
                pass

        return self.send_email(do_email, subject, body_text, body_html)

    def notify_job_submitted_to_ds(
        self,
        ds_email: str,
        job_name: str,
        do_email: str,
    ) -> SendResult:
        """Notify DS that their job was submitted and is pending review."""
        subject = f"Job Submitted: {job_name}"
        body_text = f"""Your job has been submitted!

Job: {job_name}
Submitted To: {do_email}

Your job is now pending review by the data owner.
You'll receive follow-up emails when the job is reviewed and executed.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/job_submitted_ds.html",
                    {
                        "job_name": job_name,
                        "do_email": do_email,
                    },
                )
            except Exception:
                pass

        return self.send_email(ds_email, subject, body_text, body_html)

    def notify_job_approved(
        self,
        ds_email: str,
        job_name: str,
        job_url: Optional[str] = None,
    ) -> SendResult:
        """Notify DS that their job was approved."""
        subject = f"Job Approved: {job_name}"
        body_text = f"""Your job has been approved!

Job: {job_name}

The data owner has reviewed and approved your job request.
Your job will be executed soon.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/job_approved.html",
                    {
                        "job_name": job_name,
                        "job_url": job_url,
                    },
                )
            except Exception:
                pass

        return self.send_email(ds_email, subject, body_text, body_html)

    def notify_job_executed(
        self,
        ds_email: str,
        job_name: str,
        duration: Optional[int] = None,
        results_url: Optional[str] = None,
    ) -> SendResult:
        """Notify DS that their job finished."""
        subject = f"Job Completed: {job_name}"
        body_text = f"""Your job has finished execution!

Job: {job_name}

Your job has completed successfully. Results are available.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/job_executed.html",
                    {
                        "job_name": job_name,
                        "duration": duration,
                        "results_url": results_url,
                    },
                )
            except Exception:
                pass

        return self.send_email(ds_email, subject, body_text, body_html)

    def notify_new_peer_request_to_do(
        self,
        do_email: str,
        ds_email: str,
        peer_url: Optional[str] = None,
    ) -> SendResult:
        """Notify DO about a new peer request."""
        subject = f"New Peer Request from {ds_email}"
        body_text = f"""You have a new peer request in SyftBox!

From: {ds_email}

A data scientist wants to connect with you to collaborate on data projects.

Log in to SyftBox to review this peer request.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/new_peer_request.html",
                    {
                        "ds_email": ds_email,
                        "peer_url": peer_url,
                    },
                )
            except Exception:
                pass

        return self.send_email(do_email, subject, body_text, body_html)

    def notify_peer_request_sent(
        self,
        ds_email: str,
        do_email: str,
    ) -> SendResult:
        """Notify DS that their peer request was sent."""
        subject = f"Peer Request Sent to {do_email}"
        body_text = f"""Your peer request has been sent!

To: {do_email}

Your request to connect with this data owner has been sent.
You will be notified when they accept your request.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/peer_request_sent.html",
                    {
                        "do_email": do_email,
                    },
                )
            except Exception:
                pass

        return self.send_email(ds_email, subject, body_text, body_html)

    def notify_peer_added_to_ds(
        self,
        ds_email: str,
        do_email: str,
    ) -> SendResult:
        """Notify DS that they added a peer."""
        subject = f"Peer Added: {do_email}"
        body_text = f"""You successfully added a peer in SyftBox!

Peer: {do_email}

You can now submit jobs and collaborate with this data owner.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/peer_added.html",
                    {
                        "do_email": do_email,
                    },
                )
            except Exception:
                pass

        return self.send_email(ds_email, subject, body_text, body_html)

    def notify_peer_request_granted(
        self,
        ds_email: str,
        do_email: str,
    ) -> SendResult:
        """Notify DS that their peer request was accepted."""
        subject = f"Peer Request Accepted: {do_email}"
        body_text = f"""Your peer request has been accepted!

Peer: {do_email}

The data owner has accepted your request. You can now collaborate with them.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/peer_granted.html",
                    {
                        "do_email": do_email,
                    },
                )
            except Exception:
                pass

        return self.send_email(ds_email, subject, body_text, body_html)

    def notify_job_rejected_to_ds(
        self,
        ds_email: str,
        job_name: str,
        friendly_reason: str,
    ) -> SendResult:
        """Notify DS that their job was not approved."""
        subject = f"Job Not Approved: {job_name}"
        body_text = f"""Your job was not approved.

Job: {job_name}

{friendly_reason}

If you believe this is an error, please contact the data owner.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/job_rejected_ds.html",
                    {
                        "job_name": job_name,
                        "friendly_reason": friendly_reason,
                    },
                )
            except Exception:
                pass

        return self.send_email(ds_email, subject, body_text, body_html)

    # --- DO notifications for job lifecycle (threaded) ---

    def notify_job_approved_to_do(
        self,
        do_email: str,
        job_name: str,
        ds_email: str,
        thread_id: Optional[str] = None,
    ) -> SendResult:
        """Notify DO that a job was auto-approved."""
        subject = f"Re: Job: {job_name}"
        body_text = f"""Job auto-approved!

Job: {job_name}
From: {ds_email}

This job matched your approval criteria and is now running.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/job_approved_do.html",
                    {
                        "job_name": job_name,
                        "ds_email": ds_email,
                    },
                )
            except Exception:
                pass

        return self.send_email(do_email, subject, body_text, body_html, thread_id)

    def notify_job_completed_to_do(
        self,
        do_email: str,
        job_name: str,
        ds_email: str,
        duration: Optional[int] = None,
        thread_id: Optional[str] = None,
    ) -> SendResult:
        """Notify DO that a job completed."""
        subject = f"Re: Job: {job_name}"
        duration_text = f"\nDuration: {duration}s" if duration else ""
        body_text = f"""Job completed!

Job: {job_name}
From: {ds_email}{duration_text}

The job has finished execution. Results are available.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/job_completed_do.html",
                    {
                        "job_name": job_name,
                        "ds_email": ds_email,
                        "duration": duration,
                    },
                )
            except Exception:
                pass

        return self.send_email(do_email, subject, body_text, body_html, thread_id)

    def notify_job_failed_to_do(
        self,
        do_email: str,
        job_name: str,
        ds_email: str,
        error_output: Optional[str] = None,
        return_code: Optional[int] = None,
        duration: Optional[int] = None,
        thread_id: Optional[str] = None,
    ) -> SendResult:
        """Notify DO that a job failed during execution."""
        subject = f"Re: Job: {job_name}" if thread_id else f"Job Failed: {job_name}"
        rc_text = f"\nReturn Code: {return_code}" if return_code is not None else ""
        duration_text = f"\nDuration: {duration}s" if duration else ""
        body_text = f"""Job failed!

Job: {job_name}
From: {ds_email}{rc_text}{duration_text}

The job failed during execution.
"""
        if error_output:
            body_text += (
                f"\n--- Error Output ---\n{error_output}\n--- End Error Output ---\n"
            )

        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/job_failed_do.html",
                    {
                        "job_name": job_name,
                        "ds_email": ds_email,
                        "error_output": error_output,
                        "return_code": return_code,
                        "duration": duration,
                    },
                )
            except Exception:
                pass

        return self.send_email(do_email, subject, body_text, body_html, thread_id)

    def notify_job_rejected_to_do(
        self,
        do_email: str,
        job_name: str,
        ds_email: str,
        reason: str,
        thread_id: Optional[str] = None,
    ) -> SendResult:
        """Notify DO that a job was rejected."""
        subject = f"Re: Job: {job_name}"
        body_text = f"""Job rejected.

Job: {job_name}
From: {ds_email}
Reason: {reason}

This job did not match your approval criteria and was not executed.
"""
        body_html = None
        if self.use_html and self.renderer:
            try:
                body_html = self.renderer.render(
                    "emails/job_rejected_do.html",
                    {
                        "job_name": job_name,
                        "ds_email": ds_email,
                        "reason": reason,
                    },
                )
            except Exception:
                pass

        return self.send_email(do_email, subject, body_text, body_html, thread_id)
