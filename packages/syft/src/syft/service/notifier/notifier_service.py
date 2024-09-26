# stdlib
from datetime import datetime
from datetime import timedelta
import logging

# third party
from pydantic import EmailStr

# relative
from ...abstract_server import AbstractServer
from ...serde.serializable import serializable
from ...store.db.db import DBManager
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from ..context import AuthedServiceContext
from ..notification.email_templates import PasswordResetTemplate
from ..notification.notifications import Notification
from ..response import SyftSuccess
from ..service import AbstractService
from .notifier import EmailFrequency
from .notifier import NotificationPreferences
from .notifier import NotifierSettings
from .notifier import UserNotificationActivity
from .notifier_enums import EMAIL_TYPES
from .notifier_enums import NOTIFICATION_FREQUENCY
from .notifier_enums import NOTIFIERS
from .notifier_stash import NotifierStash

logger = logging.getLogger(__name__)


class RateLimitException(SyftException):
    public_message = "Rate limit exceeded."


@serializable(canonical_name="NotifierService", version=1)
class NotifierService(AbstractService):
    stash: NotifierStash

    def __init__(self, store: DBManager) -> None:
        self.stash = NotifierStash(store=store)

    @as_result(StashException)
    def settings(
        self,
        context: AuthedServiceContext,
    ) -> NotifierSettings:
        """Get Notifier Settings

        Args:
            context: The request context
        Returns:
            NotifierSettings | None: The notifier settings, if it exists; None otherwise.
        """
        return self.stash.get(credentials=context.credentials).unwrap(
            public_message="Error getting notifier settings"
        )

    def user_settings(
        self,
        context: AuthedServiceContext,
    ) -> NotificationPreferences:
        user_view = context.server.services.user.get_current_user(context)
        notifications = user_view.notifications_enabled
        return NotificationPreferences(
            email=notifications[NOTIFIERS.EMAIL],
            sms=notifications[NOTIFIERS.SMS],
            slack=notifications[NOTIFIERS.SLACK],
            app=notifications[NOTIFIERS.APP],
        )

    def _set_notifier(self, context: AuthedServiceContext, active: bool) -> SyftSuccess:
        notifier = self.stash.get(credentials=context.credentials).unwrap(
            public_message="Notifier settings not found."
        )
        notifier.active = active
        self.stash.update(credentials=context.credentials, obj=notifier).unwrap()

        active_s = "active" if active else "inactive"
        return SyftSuccess(message=f"Notifier set to {active_s}")

    def set_notifier_active_to_false(
        self, context: AuthedServiceContext
    ) -> SyftSuccess:
        """
        Essentially a duplicate of turn_off method.
        """
        notifier = self.stash.get(credentials=context.credentials).unwrap()
        notifier.active = False
        self.stash.update(credentials=context.credentials, obj=notifier).unwrap()
        return SyftSuccess(message="notifier.active set to false.")

    @as_result(SyftException)
    def turn_on(
        self,
        context: AuthedServiceContext,
        email_username: str | None = None,
        email_password: str | None = None,
        email_sender: str | None = None,
        email_server: str | None = None,
        email_port: int | None = 587,
    ) -> SyftSuccess:
        """Turn on email notifications.

        Args:
            email_username (Optional[str]): Email server username. Defaults to None.
            email_password (Optional[str]): Email email server password. Defaults to None.
            sender_email (Optional[str]): Email sender email. Defaults to None.
        Returns:
            SyftSuccess: success response.

        Raises:
            SyftException: any error that occurs during the process
        """

        # 1 -  If something went wrong at db level, return the error
        notifier = self.stash.get(credentials=context.credentials).unwrap()

        # 2 - If one of the credentials are set alone, return an error
        if (email_username and not email_password) or (
            not email_username and email_password
        ):
            raise SyftException(
                public_message="You must provide both username and password"
            )

        # 3 - If notifier doesn't have a email server / port and the user didn't provide them, return an error
        if not (email_server and email_port) and not notifier.email_server:
            raise SyftException(
                public_message="You must provide both server and port to enable notifications."
            )

        logger.debug("Got notifier from db")
        skip_auth: bool = False
        # If no new credentials provided, check for existing ones
        if not (email_username and email_password):
            if not (notifier.email_username and notifier.email_password):
                skip_auth = True
            else:
                logger.debug("No new credentials provided. Using existing ones.")
                email_password = notifier.email_password
                email_username = notifier.email_username

        valid_credentials = True
        if not skip_auth:
            valid_credentials = notifier.validate_email_credentials(
                username=email_username,
                password=email_password,
                server=email_server or notifier.email_server,
                port=email_port or notifier.email_port,
            )

        if not valid_credentials:
            logger.error("Invalid SMTP credentials.")
            raise SyftException(public_message=("Invalid SMTP credentials."))

        notifier.email_password = email_password
        notifier.email_username = email_username

        if email_server:
            notifier.email_server = email_server
        if email_port:
            notifier.email_port = email_port

        # Email sender verification
        if not email_sender and not notifier.email_sender:
            raise SyftException(
                public_message="You must provide a sender email address to enable notifications."
            )

        # If email_rate_limit isn't defined yet.
        if not notifier.email_rate_limit:
            notifier.email_rate_limit = {PasswordResetTemplate.__name__: 3}

        if email_sender:
            try:
                EmailStr._validate(email_sender)
            except ValueError:
                raise SyftException(
                    publiccmessage="Invalid sender email address. Please check your email address."
                )
            notifier.email_sender = email_sender

        notifier.active = True
        logger.debug(
            "Email credentials are valid. Updating the notifier settings in the db."
        )

        self.stash.update(credentials=context.credentials, obj=notifier).unwrap()
        context.server.services.settings.update(context, notifications_enabled=True)
        return SyftSuccess(message="Notifications enabled successfully.")

    @as_result(SyftException)
    def set_email_batch(
        self,
        context: AuthedServiceContext,
        email_type: EMAIL_TYPES,
        frequency: NOTIFICATION_FREQUENCY,
        start_time: str = "",
    ) -> SyftSuccess:
        if start_time == "" and frequency is not NOTIFICATION_FREQUENCY.INSTANT:
            raise SyftException(
                "If frequency isn't INSTANT, you must set a start time for the notifications to be dispatched."
            )

        if frequency is not NOTIFICATION_FREQUENCY.INSTANT:
            start_time = start_time.lower()
            try:
                if "pm" in start_time or "am" in start_time:
                    time_obj = datetime.strptime(start_time, "%I:%M %p")
                else:
                    time_obj = datetime.strptime(start_time, "%H:%M")
            except ValueError:
                raise SyftException(
                    "Invalid time format."
                    + "Please enter the start time in one of the following format examples:"
                    + "'14:00' or '2:00 PM'."
                )
        else:
            time_obj = datetime.now()

        notifier = self.stash.get(credentials=context.credentials).unwrap()
        notifier.email_frequency[email_type.value] = EmailFrequency(
            frequency=frequency, start_time=time_obj
        )
        self.stash.update(credentials=context.credentials, obj=notifier).unwrap()
        return SyftSuccess(message="Configuration set successfully.")

    @as_result(StashException)
    def turn_off(
        self,
        context: AuthedServiceContext,
    ) -> SyftSuccess:
        """
        Turn off email notifications service.
        PySyft notifications will still work.
        """
        notifier = self.stash.get(credentials=context.credentials).unwrap()

        notifier.active = False
        self.stash.update(credentials=context.credentials, obj=notifier).unwrap()
        context.server.services.settings.update(context, notifications_enabled=False)
        return SyftSuccess(message="Notifications disabled succesfullly")

    @as_result(SyftException)
    def activate(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS = NOTIFIERS.EMAIL
    ) -> SyftSuccess:
        """
        Activate email notifications for the authenticated user.
        This will only work if the datasite owner has enabled notifications.
        """
        return context.server.services.user.enable_notifications(
            context, notifier_type=notifier_type
        ).unwrap()

    @as_result(SyftException)
    def deactivate(
        self, context: AuthedServiceContext, notifier_type: NOTIFIERS = NOTIFIERS.EMAIL
    ) -> SyftSuccess:
        """Deactivate email notifications for the authenticated user
        This will only work if the datasite owner has enabled notifications.
        """
        result = context.server.services.user.disable_notifications(
            context, notifier_type=notifier_type
        )
        return result

    def is_time_to_dispatch(
        self, notification_frequency: EmailFrequency, current_time: datetime
    ) -> bool:
        frequency = notification_frequency.frequency
        start_time = notification_frequency.start_time

        # Define period_timedelta based on frequency
        if frequency == NOTIFICATION_FREQUENCY.INSTANT:
            return True
        if frequency == NOTIFICATION_FREQUENCY.SIX_HOURS:
            period = timedelta(hours=6)
        elif frequency == NOTIFICATION_FREQUENCY.TWELVE_HOURS:
            period = timedelta(hours=12)
        elif frequency == NOTIFICATION_FREQUENCY.DAILY:
            period = timedelta(days=1)
        elif frequency == NOTIFICATION_FREQUENCY.WEEKLY:
            period = timedelta(weeks=1)

        # Calculate how many full periods have passed since start_time
        elapsed_time = current_time - start_time
        if elapsed_time < timedelta(0):
            return False  # Current time is before the start time

        periods_elapsed = int(elapsed_time // period)
        next_dispatch_time = start_time + periods_elapsed * period

        # Allow a small margin of error (e.g., 1 minute) to account for processing delays
        margin = timedelta(minutes=1)
        time_difference = current_time - next_dispatch_time

        # Check if current_time is within the margin of the scheduled dispatch time
        return timedelta(0) <= time_difference <= margin

    @staticmethod
    @as_result(SyftException)
    def init_notifier(
        server: AbstractServer,
        email_username: str | None = None,
        email_password: str | None = None,
        email_sender: str | None = None,
        smtp_port: int | None = None,
        smtp_host: str | None = None,
    ) -> SyftSuccess | None:
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
            SyftSuccess
        """
        try:
            # Create a new NotifierStash since its a static method.
            notifier_stash = NotifierStash(store=server.db)
            should_update = False

            # Get the notifier
            # If notifier doesn't exist, create a new one
            try:
                notifier = notifier_stash.get(server.signing_key.verify_key).unwrap()
                should_update = True
            except NotFoundException:
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

                if not validation_result or sender_not_set:
                    logger.error("Notifier validation error")
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

            if should_update:
                notifier_stash.update(
                    credentials=server.signing_key.verify_key, obj=notifier
                ).unwrap()
            else:
                notifier_stash.set(server.signing_key.verify_key, notifier).unwrap()
            return SyftSuccess(
                message="Notifier initialized successfully", value=notifier
            )
        except Exception as e:
            raise SyftException.from_exception(
                e, public_message=f"Error initializing notifier. {e}"
            )

    def set_email_rate_limit(
        self, context: AuthedServiceContext, email_type: EMAIL_TYPES, daily_limit: int
    ) -> SyftSuccess:
        notifier = self.stash.get(context.credentials).unwrap(
            public_message="Couldn't set the email rate limit."
        )
        notifier.email_rate_limit[email_type.value] = daily_limit
        self.stash.update(credentials=context.credentials, obj=notifier)

        return SyftSuccess(message="Email rate limit updated!")

    # This is not a public API.
    # This method is used by other services to dispatch notifications internally
    @as_result(SyftException, RateLimitException)
    def dispatch_notification(
        self, context: AuthedServiceContext, notification: Notification
    ) -> SyftSuccess:
        admin_key = context.server.services.user.root_verify_key

        # Silently fail on notification not delivered
        try:
            notifier = self.stash.get(admin_key).unwrap(
                public_message="The mail service ran out of quota or some notifications failed to be delivered.\n"
                + "Please check the health of the mailing server."
            )
        except NotFoundException:
            logger.debug("There is no notifier service to ship the notification")
            raise SyftException(
                public_message="No notifier service to ship the notification."
            )
        except StashException as exc:
            logger.error(f"Error getting notifier settings: {exc}")
            raise SyftException(
                public_message="Failed to get notifier settings. Please check the logs."
            )

        # If notifier is active
        if notifier.active and notification.email_template is not None:
            logger.debug("Checking user email activity")

            if notifier.email_activity.get(notification.email_template.__name__, None):
                user_activity = notifier.email_activity[
                    notification.email_template.__name__
                ].get(notification.to_user_verify_key, None)

                # If there's no user activity
                if user_activity is None:
                    notifier.email_activity[notification.email_template.__name__][
                        notification.to_user_verify_key
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
                        raise RateLimitException(
                            public_message="Couldn't send the email. You have surpassed the"
                            + " email threshold limit. Please try again later."
                        )
            else:
                notifier.email_activity[notification.email_template.__name__] = {
                    notification.to_user_verify_key: UserNotificationActivity(
                        count=1, date=datetime.now()
                    )
                }

            email_frequency = notifier.email_frequency.get(
                notification.email_template.__name__,
                EmailFrequency(frequency=NOTIFICATION_FREQUENCY.INSTANT),
            )

            if email_frequency.frequency == NOTIFICATION_FREQUENCY.INSTANT:
                notifier.send_notifications(
                    context=context, notification=notification
                ).unwrap()
            else:
                queue_dict = notifier.email_queue.get(
                    notification.email_template.__name__, {}
                )
                if len(queue_dict) == 0:
                    notifier.email_queue[notification.email_template.__name__] = (
                        queue_dict
                    )

                user_queue = queue_dict.get(notification.to_user_verify_key, [])

                if len(user_queue) == 0:
                    queue_dict[notification.to_user_verify_key] = user_queue

                user_queue.append(notification)

            self.stash.update(credentials=admin_key, obj=notifier).unwrap()

        # If notifier isn't active, return None
        return SyftSuccess(message="Notification dispatched successfully")
