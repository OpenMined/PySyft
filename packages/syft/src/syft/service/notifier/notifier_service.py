# stdlib

# stdlib

# third party
from pydantic import EmailStr
from result import Err
from result import Ok
from result import Result

# relative
from ...abstract_node import AbstractNode
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..notification.notifications import Notification
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from .notifier import NotificationPreferences
from .notifier import NotifierSettings
from .notifier_enums import NOTIFIERS
from .notifier_stash import NotifierStash


@serializable()
class NotifierService(AbstractService):
    store: DocumentStore
    stash: NotifierStash  # Which stash should we use?

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = NotifierStash(store=store)

    def settings(  # Maybe just notifier.settings
        self,
        context: AuthedServiceContext,
    ) -> NotifierSettings | SyftError:
        """Get Notifier Settings

        Args:
            context: The request context
        Returns:
            Union[NotifierSettings, SyftError]: Notifier Settings or SyftError
        """
        result = self.stash.get(credentials=context.credentials)
        if result.is_err():
            return SyftError(message="Error getting notifier settings")

        return result.ok()

    def user_settings(
        self,
        context: AuthedServiceContext,
    ) -> NotificationPreferences:
        user_service = context.node.get_service("userservice")
        user_view = user_service.get_current_user(context)
        notifications = user_view.notifications_enabled
        return NotificationPreferences(
            email=notifications[NOTIFIERS.EMAIL],
            sms=notifications[NOTIFIERS.SMS],
            slack=notifications[NOTIFIERS.SLACK],
            app=notifications[NOTIFIERS.APP],
        )

    def turn_on(
        self,
        context: AuthedServiceContext,
        email_username: str | None = None,
        email_password: str | None = None,
        email_sender: str | None = None,
        email_server: str | None = None,
        email_port: int | None = 587,
    ) -> SyftSuccess | SyftError:
        """Turn on email notifications.

        Args:
            email_username (Optional[str]): Email server username. Defaults to None.
            email_password (Optional[str]): Email email server password. Defaults to None.
            sender_email (Optional[str]): Email sender email. Defaults to None.
        Returns:
            Union[SyftSuccess, SyftError]: A union type representing the success or error response.

        Raises:
            None

        """

        result = self.stash.get(credentials=context.credentials)

        # 1 -  If something went wrong at db level, return the error
        if result.is_err():
            return SyftError(message=result.err())

        # 2 - If one of the credentials are set alone, return an error
        if (
            email_username
            and not email_password
            or email_password
            and not email_username
        ):
            return SyftError(message="You must provide both username and password")

        notifier = result.ok()

        # 3 - If notifier doesn't have a email server / port and the user didn't provide them, return an error
        if not (email_server and email_port) and not notifier.email_server:
            return SyftError(
                message="You must provide both server and port to enable notifications."
            )

        print("[LOG] Got notifier from db")
        # If no new credentials provided, check for existing ones
        if not (email_username and email_password):
            if not (notifier.email_username and notifier.email_password):
                return SyftError(
                    message="No valid token has been added to the domain."
                    + "You can add a pair of SMTP credentials via "
                    + "<client>.settings.enable_notifications(email=<>, password=<>)"
                )
            else:
                print("[LOG] No new credentials provided. Using existing ones.")
                email_password = notifier.email_password
                email_username = notifier.email_username
        print("[LOG] Validating credentials...")

        validation_result = notifier.validate_email_credentials(
            username=email_username,
            password=email_password,
            server=email_server if email_server else notifier.email_server,
            port=email_port if email_port else notifier.email_port,
        )

        if validation_result.is_err():
            return SyftError(
                message="Invalid SMTP credentials. Please check your username and password."
            )

        notifier.email_password = email_password
        notifier.email_username = email_username

        if email_server:
            notifier.email_server = email_server
        if email_port:
            notifier.email_port = email_port

        # Email sender verification
        if not email_sender and not notifier.email_sender:
            return SyftError(
                message="You must provide a sender email address to enable notifications."
            )

        if email_sender:
            try:
                EmailStr._validate(email_sender)
            except ValueError:
                return SyftError(
                    message="Invalid sender email address. Please check your email address."
                )
            notifier.email_sender = email_sender

        notifier.active = True
        print(
            "[LOG] Email credentials are valid. Updating the notifier settings in the db."
        )

        result = self.stash.update(credentials=context.credentials, settings=notifier)
        if result.is_err():
            return SyftError(message=result.err())
        return SyftSuccess(message="Notifications enabled successfully.")

    def turn_off(
        self,
        context: AuthedServiceContext,
    ) -> SyftSuccess | SyftError:
        """
        Turn off email notifications service.
        PySyft notifications will still work.
        """

        result = self.stash.get(credentials=context.credentials)

        if result.is_err():
            return SyftError(message=result.err())

        notifier = result.ok()
        notifier.active = False
        result = self.stash.update(credentials=context.credentials, settings=notifier)
        if result.is_err():
            return SyftError(message=result.err())
        return SyftSuccess(message="Notifications disabled succesfullly")

    def activate(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS = NOTIFIERS.EMAIL
    ) -> SyftSuccess | SyftError:
        """
        Activate email notifications for the authenticated user.
        This will only work if the domain owner has enabled notifications.
        """

        user_service = context.node.get_service("userservice")
        return user_service.enable_notifications(context, notifier_type=notifier_type)

    def deactivate(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS = NOTIFIERS.EMAIL
    ) -> SyftSuccess | SyftError:
        """Deactivate email notifications for the authenticated user
        This will only work if the domain owner has enabled notifications.
        """

        user_service = context.node.get_service("userservice")
        return user_service.disable_notifications(context, notifier_type=notifier_type)

    @staticmethod
    def init_notifier(
        node: AbstractNode,
        email_username: str | None = None,
        email_password: str | None = None,
        email_sender: str | None = None,
        smtp_port: int | None = None,
        smtp_host: str | None = None,
    ) -> Result[Ok, Err]:
        """Initialize Notifier settings for a Node.
        If settings already exist, it will use the existing one.
        If not, it will create a new one.

        Args:
            node: Node to initialize the notifier
            active: If notifier should be active
            email_username: Email username to send notifications
            email_password: Email password to send notifications
        Raises:
            Exception: If something went wrong
        Returns:
            Union: SyftSuccess or SyftError
        """
        try:
            # Create a new NotifierStash since its a static method.
            notifier_stash = NotifierStash(store=node.document_store)
            result = notifier_stash.get(node.signing_key.verify_key)
            if result.is_err():
                raise Exception(f"Could not create notifier: {result}")

            # Get the notifier
            notifier = result.ok()
            # If notifier doesn't exist, create a new one

            if not notifier:
                notifier = NotifierSettings()
                notifier.active = False  # Default to False

            # TODO: this should be a method in NotifierSettings
            if email_username and email_password:
                validation_result = notifier.validate_email_credentials(
                    username=email_username,
                    password=email_password,
                    server=smtp_host,
                    port=smtp_port,
                )

                sender_not_set = not email_sender and not notifier.email_sender
                if validation_result.is_err() or sender_not_set:
                    print(
                        "Ops something went wrong while trying to setup your notification system.",
                        "Please check your credentials and configuration.",
                    )
                    notifier.active = False
                else:
                    notifier.email_password = email_password
                    notifier.email_username = email_username
                    notifier.email_sender = email_sender
                    notifier.email_server = smtp_host
                    notifier.email_port = smtp_port
                    notifier.active = True

            notifier_stash.set(node.signing_key.verify_key, notifier)
            return Ok("Notifier initialized successfully")

        except Exception as e:
            raise Exception(f"Error initializing notifier. \n {e}")

    # This is not a public API.
    # This method is used by other services to dispatch notifications internally
    def dispatch_notification(
        self, context: AuthedServiceContext, notification: Notification
    ) -> SyftError:
        admin_key = context.node.get_service("userservice").admin_verify_key()
        notifier = self.stash.get(admin_key)
        if notifier.is_err():
            return SyftError(
                message="The mail service ran out of quota or some notifications failed to be delivered.\n"
                + "Please check the health of the mailing server."
            )

        notifier = notifier.ok()
        # If notifier is active
        if notifier.active:
            resp = notifier.send_notifications(
                context=context, notification=notification
            )
            if resp.is_err():
                return SyftError(message=resp.err())

        # If notifier isn't active, return None
        return SyftSuccess(message="Notifications dispatched successfully")
