# stdlib
from typing import TYPE_CHECKING
from typing import cast

# relative
from ...store.linked_obj import LinkedObject
from ..context import AuthedServiceContext

if TYPE_CHECKING:
    # relative
    from .notifications import Notification


class EmailTemplate:
    @staticmethod
    def email_title(notification: "Notification", context: AuthedServiceContext) -> str:
        return ""

    @staticmethod
    def email_body(notification: "Notification", context: AuthedServiceContext) -> str:
        return ""


class OnBoardEmailTemplate(EmailTemplate):
    @staticmethod
    def email_title(notification: "Notification", context: AuthedServiceContext) -> str:
        return f"Welcome to {context.node.name} node!"

    @staticmethod
    def email_body(notification: "Notification", context: AuthedServiceContext) -> str:
        user_service = context.node.get_service("userservice")
        admin_name = user_service.get_by_verify_key(
            user_service.admin_verify_key()
        ).name

        head = (
            f"""
        <head>
            <title>Welcome to {context.node.name}</title>
        """
            + """
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    color: #333;
                    line-height: 1.6;
                }
                .container {
                    max-width: 600px;
                    margin: 20px auto;
                    padding: 20px;
                    background: #fff;
                }
                h1 {
                    color: #0056b3;
                }
                .feature {
                    background-color: #e7f1ff;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                }
                .footer {
                    text-align: center;
                    font-size: 14px;
                    color: #aaa;
                }
            </style>
        </head>
        """
        )

        body = f"""
        <body>
            <div class="container">
                <h1>Welcome to {context.node.name} node!</h1>
                <p>Hello,</p>
                <p>We're thrilled to have you on board and
                excited to help you get started with our powerful features:</p>

                <div class="feature">
                    <h3>Remote Data Science</h3>
                    <p>Access and analyze data from anywhere, using our comprehensive suite of data science tools.</p>
                </div>

                <div class="feature">
                    <h3>Remote Code Execution</h3>
                    <p>Execute code remotely on private data, ensuring flexibility and efficiency in your research.</p>
                </div>

                <!-- Add more features here if needed -->

                <p>Explore these features and much more within your account.
                If you have any questions or need assistance, don't hesitate to reach out.</p>

                <p>Cheers,</p>
                <p>{admin_name}</p>

                <div class="footer">
                    This is an automated message, please do not reply directly to this email. <br>
                    For assistance, please contact our support team.
                </div>
            </div>
        </body>
        """
        return f"""<html>{head} {body}</html>"""


class RequestEmailTemplate(EmailTemplate):
    @staticmethod
    def email_title(notification: "Notification", context: AuthedServiceContext) -> str:
        notification.linked_obj = cast(LinkedObject, notification.linked_obj)
        request_obj = notification.linked_obj.resolve_with_context(context=context).ok()

        return f"Domain {context.node.name}: A New Request ({str(request_obj.id)[:4]}) has been received!"

    @staticmethod
    def email_body(notification: "Notification", context: AuthedServiceContext) -> str:
        notification.linked_obj = cast(LinkedObject, notification.linked_obj)
        request_obj = notification.linked_obj.resolve_with_context(context=context).ok()

        head = """
        <head>
            <title>Access Request Notification</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    color: #333;
                    padding: 20px;
                }
                .container {
                    max-width: 600px;
                    margin: 0 auto;
                    background: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                .header {
                    font-size: 24px;
                    color: #333;
                    text-align: center;
                }
                .content {
                    font-size: 16px;
                    line-height: 1.6;
                }

                .request-card {
                    background-color: #ffffff;
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin-top: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }

                .request-header {
                    font-size: 18px;
                    color: #333;
                    margin-bottom: 10px;
                    font-weight: bold;
                }

                .request-content {
                    font-size: 14px;
                    line-height: 1.5;
                    color: #555;
                }
                .badge {
                    padding: 4px 10px;
                    border-radius: 15px;
                    color: white;
                    font-weight: bold;
                    margin: 10px;
                    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
                    text-transform: uppercase;
                }

                .yellow {
                    background-color: #fdd835;
                }

                .green {
                    background-color: #6dbf67;
                }

                .red {
                    background-color: #f7786b;
                }

                .button {
                    display: block;
                    width: max-content;
                    background-color: #007bff;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-color: white;
                    border-radius: 5px;
                    text-decoration: none;
                    font-weight: bold;
                    margin: 20px auto;
                }
                .footer {
                    text-align: center;
                    font-size: 14px;
                    color: #aaa;
                }
            </style>
        </head>"""

        body = f"""
        <body>
            <div class="container">
                <div class="header">
                    Request Notification
                </div>
                <div class="content">
                    <p>Hello,</p>
                    <p>A new request has been submitted and requires your attention.
                    Please review the details below:</p>

                    <div class="request-card">
                        <div class="request-header">Request Details</div>
                        <div class="request-content">

                            <p><strong>ID:</strong> {request_obj.id}</p>
                            <p>
                            <strong>Submitted By:</strong>
                            {request_obj.requesting_user_name} {request_obj.requesting_user_email or ""}
                            </p>
                            <p><strong>Date:</strong> {request_obj.request_time}</p>
                        <div style="display: flex"><p><strong>Status:</strong><div class="badge yellow">{
                            request_obj.status.name
                        }</div></div>
                        <p><strong>Changes:</strong>{
                            ",".join([change.__class__.__name__ for change in request_obj.changes])
                        }</p>
                        </div>
                    </div>
                    <p>If you did not expect this request or have concerns about it,
                    please contact our support team immediately.</p>
                </div>
                <div class="footer">
                    This is an automated message, please do not reply directly to this email. <br>
                    For assistance, please contact our support team.
                </div>
            </div>
        </body>
        """
        return f"""<html>{head} {body}</html>"""


class RequestUpdateEmailTemplate(EmailTemplate):
    @staticmethod
    def email_title(notification: "Notification", context: AuthedServiceContext) -> str:
        return f"Domain {context.node.name}: {notification.subject}"

    @staticmethod
    def email_body(notification: "Notification", context: AuthedServiceContext) -> str:
        notification.linked_obj = cast(LinkedObject, notification.linked_obj)
        request_obj = notification.linked_obj.resolve_with_context(context=context).ok()
        badge_color = "red" if request_obj.status.name == "REJECTED" else "green"
        head = """
        <head>
            <title>Access Request Notification</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    color: #333;
                    padding: 20px;
                }
                .container {
                    max-width: 600px;
                    margin: 0 auto;
                    background: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                .header {
                    font-size: 24px;
                    color: #333;
                    text-align: center;
                }
                .content {
                    font-size: 16px;
                    line-height: 1.6;
                }

                .request-card {
                    background-color: #ffffff;
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin-top: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }

                .request-header {
                    font-size: 18px;
                    color: #333;
                    margin-bottom: 10px;
                    font-weight: bold;
                }

                .badge {
                    padding: 4px 10px;
                    border-radius: 15px;
                    color: white;
                    font-weight: bold;
                    margin: 10px;
                    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
                    text-transform: uppercase;
                }

                .yellow {
                    background-color: #fdd835;
                }

                .green {
                    background-color: #6dbf67;
                }

                .red {
                    background-color: #f7786b;
                }

                .request-content {
                    font-size: 14px;
                    line-height: 1.5;
                    color: #555;
                }

                .button {
                    display: block;
                    width: max-content;
                    background-color: #007bff;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-color: white;
                    border-radius: 5px;
                    text-decoration: none;
                    font-weight: bold;
                    margin: 20px auto;
                }
                .footer {
                    text-align: center;
                    font-size: 14px;
                    color: #aaa;
                }
            </style>
        </head>"""

        body = f"""
        <body>
            <div class="container">
                <div class="header">
                    Request Notification
                </div>
                <div class="content">
                    <p>Hello,</p>
                    <p>The status of your recent request has been updated.
                    Below is the latest information regarding it:</p>

                    <div class="request-card">
                        <div class="request-header">Request Details</div>
                        <div class="request-content">

                            <p><strong>ID:</strong> {request_obj.id}</p>
                            <p>
                            <strong>Submitted By:</strong>
                            {request_obj.requesting_user_name} {request_obj.requesting_user_email or ""}
                            </p>
                            <p><strong>Date:</strong> {request_obj.request_time}</p>
                            <div style="display: flex"><p><strong>Status:</strong><div class="badge {badge_color}">{
                                request_obj.status.name
                            }</div></div>
                            <p>
                            <strong>Changes:</strong>
                            {",".join([change.__class__.__name__ for change in request_obj.changes])}
                            </p>
                        </div>
                    </div>
                    <p>If you did not expect this request or have concerns about it,
                    please contact our support team immediately.</p>
                </div>
                <div class="footer">
                    This is an automated message, please do not reply directly to this email. <br>
                    For assistance, please contact our support team.
                </div>
            </div>
        </body>
        """
        return f"""<html>{head} {body}</html>"""
