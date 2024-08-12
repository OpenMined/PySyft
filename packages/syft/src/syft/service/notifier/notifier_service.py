# stdlib
from datetime import datetime
import logging
import traceback

# third party
from pydantic import EmailStr
from result import Err
from result import Ok
from result import Result

# relative
from ...abstract_server import AbstractServer
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ..context import AuthedServiceContext
from ..notification.email_templates import PasswordResetTemplate
from ..notification.notifications import Notification
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from .notifier import NotificationPreferences
from .notifier import NotifierSettings
from .notifier import UserNotificationActivity
from .notifier_enums import EMAIL_TYPES
from .notifier_enums import NOTIFIERS
from .notifier_stash import NotifierStash

logger = logging.getLogger(__name__)


@serializable(canonical_name="NotifierService", version=1)
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
        user_service = context.server.get_service("userservice")
        user_view = user_service.get_current_user(context)
        notifications = user_view.notifications_enabled
        return NotificationPreferences(
            email=notifications[NOTIFIERS.EMAIL],
            sms=notifications[NOTIFIERS.SMS],
            slack=notifications[NOTIFIERS.SLACK],
            app=notifications[NOTIFIERS.APP],
        )

    def set_notifier_active_to_true(
        self, context: AuthedServiceContext
    ) -> SyftSuccess | SyftError:
        result = self.stash.get(credentials=context.credentials)
        if result.is_err():
            return SyftError(message=result.err())

        notifier = result.ok()
        if notifier is None:
            return SyftError(message="Notifier settings not found.")
        notifier.active = True
        result = self.stash.update(credentials=context.credentials, settings=notifier)
        if result.is_err():
            return SyftError(message=result.err())
        return SyftSuccess(message="notifier.active set to true.")

    def set_notifier_active_to_false(
        self, context: AuthedServiceContext
    ) -> SyftSuccess:
        """
        Essentially a duplicate of turn_off method.
        """
        result = self.stash.get(credentials=context.credentials)
        if result.is_err():
            return SyftError(message=result.err())

        notifier = result.ok()
        if notifier is None:
            return SyftError(message="Notifier settings not found.")

        notifier.active = False
        result = self.stash.update(credentials=context.credentials, settings=notifier)
        if result.is_err():
            return SyftError(message=result.err())
        return SyftSuccess(message="notifier.active set to false.")

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

        logging.debug("Got notifier from db")
        # If no new credentials provided, check for existing ones
        if not (email_username and email_password):
            if not (notifier.email_username and notifier.email_password):
                return SyftError(
                    message="No valid token has been added to the datasite."
                    + "You can add a pair of SMTP credentials via "
                    + "<client>.settings.enable_notifications(email=<>, password=<>)"
                )
            else:
                logging.debug("No new credentials provided. Using existing ones.")
                email_password = notifier.email_password
                email_username = notifier.email_username

        validation_result = notifier.validate_email_credentials(
            username=email_username,
            password=email_password,
            server=email_server if email_server else notifier.email_server,
            port=email_port if email_port else notifier.email_port,
        )

        if validation_result.is_err():
            logging.error(f"Invalid SMTP credentials {validation_result.err()}")
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

        # If email_rate_limit isn't defined yet.
        if not notifier.email_rate_limit:
            notifier.email_rate_limit = {PasswordResetTemplate.__name__: 3}

        if email_sender:
            try:
                EmailStr._validate(email_sender)
            except ValueError:
                return SyftError(
                    message="Invalid sender email address. Please check your email address."
                )
            notifier.email_sender = email_sender

        notifier.active = True
        logging.debug(
            "Email credentials are valid. Updating the notifier settings in the db."
        )

        result = self.stash.update(credentials=context.credentials, settings=notifier)
        if result.is_err():
            return SyftError(message=result.err())

        settings_service = context.server.get_service("settingsservice")
        result = settings_service.update(context, notifications_enabled=True)
        if isinstance(result, SyftError):
            logger.info(f"Failed to update Server Settings: {result.message}")

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

        settings_service = context.server.get_service("settingsservice")
        result = settings_service.update(context, notifications_enabled=False)
        if isinstance(result, SyftError):
            logger.info(f"Failed to update Server Settings: {result.message}")

        return SyftSuccess(message="Notifications disabled succesfullly")

    def activate(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS = NOTIFIERS.EMAIL
    ) -> SyftSuccess | SyftError:
        """
        Activate email notifications for the authenticated user.
        This will only work if the datasite owner has enabled notifications.
        """

        user_service = context.server.get_service("userservice")
        return user_service.enable_notifications(context, notifier_type=notifier_type)

    def deactivate(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS = NOTIFIERS.EMAIL
    ) -> SyftSuccess | SyftError:
        """Deactivate email notifications for the authenticated user
        This will only work if the datasite owner has enabled notifications.
        """

        user_service = context.server.get_service("userservice")
        return user_service.disable_notifications(context, notifier_type=notifier_type)

    @staticmethod
    def init_notifier(
        server: AbstractServer,
        email_username: str | None = None,
        email_password: str | None = None,
        email_sender: str | None = None,
        smtp_port: int | None = None,
        smtp_host: str | None = None,
    ) -> Result[Ok, Err]:
        """Initialize Notifier settings for a Server.
        If settings already exist, it will use the existing one.
        If not, it will create a new one.

        Args:
            server: Server to initialize the notifier
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
            notifier_stash = NotifierStash(store=server.document_store)
            result = notifier_stash.get(server.signing_key.verify_key)
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
                    logger.error(
                        f"Notifier validation error - {validation_result.err()}.",
                    )
                    notifier.active = False
                else:
                    notifier.email_password = email_password
                    notifier.email_username = email_username
                    notifier.email_sender = email_sender
                    notifier.email_server = smtp_host
                    notifier.email_port = smtp_port
                    # Default daily email rate limit per user
                    notifier.email_rate_limit = {PasswordResetTemplate.__name__: 3}
                    notifier.active = True

            notifier_stash.set(server.signing_key.verify_key, notifier)
            return Ok("Notifier initialized successfully")

        except Exception:
            raise Exception(f"Error initializing notifier. \n {traceback.format_exc()}")

    def set_email_rate_limit(
        self, context: AuthedServiceContext, email_type: EMAIL_TYPES, daily_limit: int
    ) -> SyftSuccess | SyftError:
        notifier = self.stash.get(context.credentials)
        if notifier.is_err():
            return SyftError(message="Couldn't set the email rate limit.")

        notifier = notifier.ok()

        notifier.email_rate_limit[email_type.value] = daily_limit
        result = self.stash.update(credentials=context.credentials, settings=notifier)
        if result.is_err():
            return SyftError(message="Couldn't update the notifier.")

        return SyftSuccess(message="Email rate limit updated!")

    # This is not a public API.
    # This method is used by other services to dispatch notifications internally
    def dispatch_notification(
        self, context: AuthedServiceContext, notification: Notification
    ) -> SyftError:
        admin_key = context.server.get_service("userservice").admin_verify_key()
        notifier = self.stash.get(admin_key)
        if notifier.is_err():
            return SyftError(
                message="The mail service ran out of quota or some notifications failed to be delivered.\n"
                + "Please check the health of the mailing server."
            )

        notifier = notifier.ok()
        # If notifier is active
        if notifier.active and notification.email_template is not None:
            logging.debug("Checking user email activity")
            if notifier.email_activity.get(notification.email_template.__name__, None):
                user_activity = notifier.email_activity[
                    notification.email_template.__name__
                ].get(notification.to_user_verify_key, None)
                # If there's no user activity
                if user_activity is None:
                    notifier.email_activity[notification.email_template.__name__][
                        notification.to_user_verify_key, None
                    ] = UserNotificationActivity(count=1, date=datetime.now())
                else:  # If there's a previous user activity
                    current_state: UserNotificationActivity = notifier.email_activity[
                        notification.email_template.__name__
                    ][notification.to_user_verify_key]
                    date_refresh = abs(datetime.now() - current_state.date).days > 1

                    limit = notifier.email_rate_limit.get(
                        notification.email_template.__name__, 0
                    )
                    still_in_limit = current_state.count < limit
                    # Time interval reseted.
                    if date_refresh:
                        current_state.count = 1
                        current_state.date = datetime.now()
                    # Time interval didn't reset yet.
                    elif still_in_limit or not limit:
                        current_state.count += 1
                        current_state.date = datetime.now()
                    else:
                        return SyftError(
                            message="Couldn't send the email. You have surpassed the"
                            + " email threshold limit. Please try again later."
                        )
            else:
                notifier.email_activity[notification.email_template.__name__] = {
                    notification.to_user_verify_key: UserNotificationActivity(
                        count=1, date=datetime.now()
                    )
                }

            result = self.stash.update(credentials=admin_key, settings=notifier)
            if result.is_err():
                return SyftError(message="Couldn't update the notifier.")

            resp = notifier.send_notifications(
                context=context, notification=notification
            )
            if resp.is_err():
                return SyftError(message=resp.err())

        # If notifier isn't active, return None
        return SyftSuccess(message="Notifications dispatched successfully")
