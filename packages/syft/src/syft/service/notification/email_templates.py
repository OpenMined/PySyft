from ...abstract_node import AbstractNode
from ..context import AuthedServiceContext

class EmailTemplate:
    pass

class SuspiciousActivityEmailTemplate(EmailTemplate):

    @staticmethod
    def email_title(notification: "Notification", context: AuthedServiceContext) -> str:
        return f"Domain {context.node.name}: Suspicious Activity Detected!"
    
    @staticmethod
    def email_body(notification: "Notification", context: AuthedServiceContext) -> str:
        user_service = context.node.get_service("userservice")
        user = notification.linked_obj.resolve_with_context(
            context=context
        ).ok()

        head = """
        <head>
            <title>Suspicious Activity Alert</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    color: #333;
                    line-height: 1.6;
                }
                .container {
                    width: 80%;
                    margin: auto;
                    background: #fff;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                h1 {
                    color: #ff0000;
                }
                .alert-details {
                    margin-top: 20px;
                }
                .footer {
                    text-align: center;
                    font-size: 14px;
                    color: #aaa;
                }
            </style>
        </head>
        """

        body = f"""
        <body>
            <div class="container">
                <h1>Suspicious Activity Detected</h1>
                <p>Hello Admin,</p>
                <p>We have detected some suspicious activities in your domain node. Please review the following details:</p>
                <div class="alert-details">
                    <p><strong>Date and Time of Activity:</strong> {notification.created_at}</p>
                    <p><strong>Type of Activity:</strong> Unauthorized Register Attempt </p>
                    <p><strong>User :</strong> {user.name} {user.email}</p>
                </div>
                <p>We recommend you to take the following actions:</p>
                <ul>
                    <li>Identify the person who tried it.</li>
                    <li>Investigate his/her reasons.</li>
                    <li>Contact support if you notice any unfamiliar activity.</li>
                </ul>
                <p>Stay Safe,<br>Your {context.node.name} Team</p>
            </div>
            <div class="footer">
                This is an automated message, please do not reply directly to this email. <br>
                For assistance, please contact our support team.
            </div>
        </body>
        """
        return f"""<html>{head} {body}</html>"""

class RequestEmailTemplate(EmailTemplate):

    @staticmethod
    def email_title(notification: "Notification", context: AuthedServiceContext) -> str:
        return f"Domain {context.node.name}: New Request!"
        
    @staticmethod
    def email_body(notification: "Notification", context: AuthedServiceContext) -> str:
        request_obj = notification.linked_obj.resolve_with_context(
            context=context
        ).ok()

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
                    <p>A new request has been submitted and requires your attention. Please review the details below:</p>

                    <div class="request-card">
                        <div class="request-header">Request Details</div>
                        <div class="request-content">

                            <p><strong>ID:</strong> {request_obj.id}</p>
                            <p><strong>Submitted By:</strong> {request_obj.requesting_user_name} {request_obj.requesting_user_email or ""}</p>
                            <p><strong>Date:</strong> {request_obj.request_time}</p>
                            <p><strong>Changes:</strong> {",".join([change.__class__.__name__ for change in request_obj.changes])}</p>
                        </div>
                    </div> 
                   <p>To review and respond to this request, please click the button below:</p>
                    <a href="#" class="button">Review Request</a>
                    <p>If you did not expect this request or have concerns about it, please contact our support team immediately.</p>
                </div>
                <div class="footer">
                    This is an automated message, please do not reply directly to this email. <br>
                    For assistance, please contact our support team.
                </div>
            </div>
        </body>
        """
        return f"""<html>{head} {body}</html>"""
