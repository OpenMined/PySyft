# Option A: INTERACTS DIRECTLY WITH THE DATABASE / sql-alchemy - possibly dangerous?
# Option B: Interacts with Domain object by sending Message objects to .recv_msg_with_reply()
# Option C: The web routes have their own DomainClient() object they used to talk to the Domain object.