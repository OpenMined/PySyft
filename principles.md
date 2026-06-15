# Syft Client

Syft client is a high level client object bundling modular components which enable a user to execute map/reduce bash scripts (and any file-based resources those bash scripts might require) across a decentralized, peer-to-peer network of computers... bash scripts which are designed and tested using local mock versions of the remote/private files located on those computers... connected through whichever transport layers any two connected organizations already trust (e.g. Google Drive, Dropbox, Microsoft 365, etc.).

# Principles

## 1. File first

> 📜 State is first and foremost described by files, which are synced amongst peers on the network to communicate updates to state. Client-side state can be made directly viewable through client side applications which display the local / client-side files in a convenient manner (but not files which aren't available to the local filesystem... unless one is submitting a job to someone else to query their state). State can also be locally stored/cached in more performant ways (e.g. client or server side databases, indexes, etc.) but this is secondary for performance, and is an optional part of the system. All state is first-and-foremost made available as a file (future development may explore local, in-memory file storage...akin to the relationship between Hadoop's HDFS and Spark's upgrade to in-memory storage over HDFS, but this would still adhere to the same hierarchical addressing system of folders/files.).

## 2. File-permission-first, job-policy second, nothing third

> 📜 Access-control is first and foremost described by permissioned access to files on the system. Files are made available between users at a low level, and when file permission changes are to be requested, or when other access is to be granted, that is to be managed by the job policy framework. No other permission system exists outside of file permissions or job policies (including manual code review policy).

## 3. Offline-first

> 📜 Datasites (i.e. computers/users/peers) in the syft network can go offline/online as desired and all functionality continues to work, as messages from that server are cached locally when it is offline, and messages to each server are cached in the transport layers (e.g. Google Drive) until that datasite comes back online. Datasites can use faster, ephemeral transport layers when datasites are online at the same time (e.g. WebRTC), but this is an optional upgrade, not the foundation to the system.

## 4. Shell-first

> 📜 Functions are first and foremost described as shell scripts (run.sh) inside of a folder of resources (job_23j3ijgw/).

## 5. Schema-last

> 📜 For the core syft protocol, we seek to only require schemas or dependencies which all computers in the world are highly likely to already have (e.g. some kind of local filesystem + some kind of local shell + internet access)... with anything else that might be built on top of this core layer left open for users to steward.

## 6. Fail-softly

> 📜 When a client seeks to use state owned by another computer but the recipient isn't storing data in the way the client wishes... the server doesn't reject the job entirely (e.g. as it would if you tried to call an API endpoint that doesn't exist). Instead, the shell script which was attempted generates some error which (with the data owners's permission) the data scientist can then view to debug what's going wrong. (see client-first-work)

## 7. Manual-review-first

> 📜 We assume that all shell scripts sent to a data owner will be manually reviewed unless the data owner happens to have a policy which can automatically approve it.

## 8. DataScientist-first-debugging

> 📜 When something goes wrong... we build tools which first-and-foremost enable the client (data scientist) to be the one doing the work to come up with a solution (as opposed to all the world's data owners needing to collaborate together... such as to normalize their data together).

## 9. Peer-first

> 📜 Discoverability on the syft network is assumed to happen somewhere else (e.g. companion websites like SyftHub). The syft protocol is like Signal in this way... if they're not a contact in the network you have explicitly authorized... you don't know they exist and nobody else outside your contact list knows you exist on the network.

## 10. Modular-first

> 📜 When in doubt, we separate into optional modules such that upgrades to one doesn't require upgrades to the rest of the system.

## 11. MapReduce-first

> 📜 All interactions between data scientist and data owner personas are first-and-foremost viewed through the lens of a MapReduce system.

## 12. Unopinionated-first — Convenient-second

> 📜 Unless we need to express an opinion on how the users should do things, we seek for the core layers to be completely unopinionated (e.g. a glorified email inbox for bash scripts and supporting files), but then upon that highly unopinionated core, we add optional convenience layers to make certain actions easier for the end user.

## 13. Transport-agnostic

> 📜 Nothing about the syft protocol requires the user of any particular transport layer. And any means of getting strings from one (uniquely addressed) user to another (uniquely addressed) user are viable transport layers for the syft network.

## 14. Transport-based-auth first — any auth second

> 📜 A user on the system is marked by "a channel I can send messages to". Consequently, we bootstrap identity using the authentication of existing transport layers, but this can be used to exchange keys which might have come through some offline process, and can thus protect transport layers which might not have formal authentication (e.g. insecure transport layers).

## 15. Mock-always

> 📜 Every piece of state in the syft ecosystem comes with a mock enabling others to leverage that state in a job if they have permission to see the mock.

## 16. Automock-first

> 📜 Mock generation should be automatic as the primary way of doing things, with fallbacks to user-generated/helped mocks when privacy norms are unclear, and with no mocks in some specific instances (but this should be rare).

## 17. Single-gateway only

> 📜 There is only one job queue which sends and receives information to/from each datasite, enabling the data owner to collaborate with confidence that they can fully see anything and everything which enters to be run / exists to be disclosed from/to peers.

## 18. Job-only

> 📜 Every request by one user for another user to run computation is done through a job (including one user asking another user for updated information about state on their local machine). High level syntactic sugar might simplify this experience, but under the hood there is only one mechanism for one user to ask another user to run a computation. (this includes anything that might look or feel RPC-esque... under the hood it's also a job)
