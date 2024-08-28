class MockSMTP:
    def __init__(self, smtp_server, smtp_port, timeout):
        self.sent_mail = []
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.timeout = timeout

    def sendmail(self, from_addr, to_addrs, msg):
        self.sent_mail.append((from_addr, to_addrs, msg))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def ehlo(self):
        return True

    def has_extn(self, extn):
        return True

    def login(self, username, password):
        return True

    def starttls(self):
        return True
